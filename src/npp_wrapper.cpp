#include <algorithm>
#include <cctype>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <nppi.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

enum class Layout { HWC, CHW, SINGLE };

static void cuda_check(cudaError_t e, const char* msg){
    if (e != cudaSuccess) throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(e));
}
static void npp_check(NppStatus s, const char* msg){
    if (s != NPP_SUCCESS) throw std::runtime_error(std::string(msg) + ": NPP error code " + std::to_string(s));
}

static const char* layout_to_string(Layout l) {
    switch (l) {
        case Layout::HWC: return "HWC";
        case Layout::CHW: return "CHW";
        case Layout::SINGLE: return "SINGLE";
    }
    return "UNKNOWN";
}

static NppiInterpolationMode to_nppi_inter(const std::string& inter){
    if (inter == "nearest") return NPPI_INTER_NN;
    if (inter == "bilinear")  return NPPI_INTER_LINEAR;
    if (inter == "cubic")   return NPPI_INTER_CUBIC;
    throw std::invalid_argument("Unknown interpolation mode: " + inter);
}

static Layout determine_layout(const py::buffer_info& info,
                               std::string requested_layout,
                               int& in_h,
                               int& in_w,
                               int& channels)
{
    channels = 1;
    if (info.ndim == 2) {
        in_h = static_cast<int>(info.shape[0]);
        in_w = static_cast<int>(info.shape[1]);
        channels = 1;
        return Layout::SINGLE;
    }

    std::transform(requested_layout.begin(), requested_layout.end(), requested_layout.begin(),
                   [](unsigned char c){ return static_cast<char>(std::tolower(c)); });

    auto choose_hwc = [&]() {
        in_h = static_cast<int>(info.shape[0]);
        in_w = static_cast<int>(info.shape[1]);
        channels = static_cast<int>(info.shape[2]);
        return Layout::HWC;
    };
    auto choose_chw = [&]() {
        channels = static_cast<int>(info.shape[0]);
        in_h = static_cast<int>(info.shape[1]);
        in_w = static_cast<int>(info.shape[2]);
        return Layout::CHW;
    };

    if (requested_layout == "hwc") {
        return choose_hwc();
    }
    if (requested_layout == "chw") {
        return choose_chw();
    }
    if (requested_layout == "auto" || requested_layout.empty()) {
        const bool last_dim_small = info.shape[2] <= 4;
        const bool first_dim_small = info.shape[0] <= 4;
        if (first_dim_small && !last_dim_small) {
            return choose_chw();
        }
        if (last_dim_small && !first_dim_small) {
            return choose_hwc();
        }
        if (last_dim_small && first_dim_small) {
            return choose_hwc();
        }
        return choose_chw();
    }
    throw std::invalid_argument("layout must be 'auto', 'hwc', or 'chw'");
}

py::array_t<float> resize_32f(py::array_t<float, py::array::c_style | py::array::forcecast> img,
                              int out_h, int out_w,
                              std::string inter_str = "bilinear",
                              std::string layout = "auto")
{
    if (out_h <= 0 || out_w <= 0) throw std::invalid_argument("out_h/out_w must be > 0");
    if (img.ndim() != 2 && img.ndim() != 3) {
        throw std::invalid_argument("img must be 2D (H,W) or 3D (H,W,C)/(C,H,W) float32");
    }

    std::cout << "[npp_wrapper] resize_32f start: in_ndim=" << img.ndim()
              << ", out_h=" << out_h << ", out_w=" << out_w
              << ", inter='" << inter_str << "', layout='" << layout << "'\n";

    const auto info = img.request();
    const float* h_in = static_cast<const float*>(info.ptr);

    int in_h = 0;
    int in_w = 0;
    int channels = 1;
    Layout actual_layout = determine_layout(info, layout, in_h, in_w, channels);

    if (channels <= 0) throw std::invalid_argument("channel dimension must be > 0");

    const size_t total_elems = static_cast<size_t>(info.size);

    std::cout << "[npp_wrapper] detected layout=" << layout_to_string(actual_layout)
              << ", in_h=" << in_h << ", in_w=" << in_w
              << ", channels=" << channels << '\n';

    if (actual_layout == Layout::HWC) {
        if (!(channels == 1 || channels == 3 || channels == 4)) {
            throw std::invalid_argument("Only 1, 3, or 4 interleaved channels are supported; specify layout='chw' for planar data");
        }
    }

    const size_t bytes_in = total_elems * sizeof(float);

    // Device pointers
    float *d_in = nullptr, *d_out = nullptr;
    size_t bytes_out = static_cast<size_t>(channels) * static_cast<size_t>(out_h) * static_cast<size_t>(out_w) * sizeof(float);

    std::cout << "[npp_wrapper] cudaMalloc d_in=" << bytes_in
              << " bytes, d_out=" << bytes_out << " bytes\n";

    cuda_check(cudaMalloc(&d_in,  bytes_in),  "cudaMalloc d_in");
    cuda_check(cudaMalloc(&d_out, bytes_out), "cudaMalloc d_out");

    std::cout << "[npp_wrapper] cudaMemcpy H2D\n";
    cuda_check(cudaMemcpy(d_in, h_in, bytes_in, cudaMemcpyHostToDevice), "cudaMemcpy H2D");

    // Prepare output array
    py::array_t<float> out;
    if (actual_layout == Layout::SINGLE) {
        out = py::array_t<float>({out_h, out_w});
    } else if (actual_layout == Layout::HWC) {
        out = py::array_t<float>({out_h, out_w, channels});
    } else {
        out = py::array_t<float>({channels, out_h, out_w});
    }
    auto out_info = out.request();
    float* h_out = static_cast<float*>(out_info.ptr);

    // NPP ROI structs
    NppiSize srcSize { in_w,  in_h  };
    NppiRect srcROI  { 0, 0, in_w,  in_h  };
    NppiSize dstSize { out_w, out_h };
    NppiRect dstROI  { 0, 0, out_w, out_h };

    const auto inter_mode = to_nppi_inter(inter_str);

    auto call_resize_c1 = [&](const float* src_ptr, int src_step, float* dst_ptr, int dst_step) {
        npp_check(nppiResize_32f_C1R(
            src_ptr, src_step, srcSize, srcROI,
            dst_ptr, dst_step, dstSize, dstROI,
            inter_mode
        ), "nppiResize_32f_C1R");
        std::cout << "[npp_wrapper] nppiResize_32f_C1R call completed (step=" << src_step << ")\n";
    };

    if (actual_layout == Layout::SINGLE) {
        const int in_step = in_w * static_cast<int>(sizeof(float));
        const int out_step = out_w * static_cast<int>(sizeof(float));
        call_resize_c1(d_in, in_step, d_out, out_step);
    } else if (actual_layout == Layout::CHW) {
        const int in_step = in_w * static_cast<int>(sizeof(float));
        const int out_step = out_w * static_cast<int>(sizeof(float));
        const size_t in_plane = static_cast<size_t>(in_h) * static_cast<size_t>(in_w);
        const size_t out_plane = static_cast<size_t>(out_h) * static_cast<size_t>(out_w);
        for (int c = 0; c < channels; ++c) {
            std::cout << "[npp_wrapper] resizing channel " << c << " (CHW)\n";
            const float* src_plane = d_in + c * in_plane;
            float* dst_plane = d_out + c * out_plane;
            call_resize_c1(src_plane, in_step, dst_plane, out_step);
        }
    } else { // HWC
        const int in_step = in_w * channels * static_cast<int>(sizeof(float));
        const int out_step = out_w * channels * static_cast<int>(sizeof(float));
        switch (channels) {
            case 1:
                call_resize_c1(d_in, in_step, d_out, out_step);
                break;
            case 3:
                std::cout << "[npp_wrapper] nppiResize_32f_C3R\n";
                npp_check(nppiResize_32f_C3R(
                    d_in, in_step, srcSize, srcROI,
                    d_out, out_step, dstSize, dstROI,
                    inter_mode
                ), "nppiResize_32f_C3R");
                break;
            case 4:
                std::cout << "[npp_wrapper] nppiResize_32f_C4R\n";
                npp_check(nppiResize_32f_C4R(
                    d_in, in_step, srcSize, srcROI,
                    d_out, out_step, dstSize, dstROI,
                    inter_mode
                ), "nppiResize_32f_C4R");
                break;
        }
    }

    // Copy back
    std::cout << "[npp_wrapper] cudaMemcpy D2H\n";
    cuda_check(cudaMemcpy(h_out, d_out, bytes_out, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");

    cudaFree(d_in);
    cudaFree(d_out);

    std::cout << "[npp_wrapper] resize_32f done\n";

    return out;
}

py::array_t<float> resize_sqr_pixel_32f(py::array_t<float, py::array::c_style | py::array::forcecast> img,
                                        int out_h, int out_w,
                                        std::optional<double> scale_y_opt = std::nullopt,
                                        std::optional<double> scale_x_opt = std::nullopt,
                                        double shift_y = 0.0,
                                        double shift_x = 0.0,
                                        std::string inter_str = "bilinear",
                                        std::string layout = "auto")
{
    if (out_h <= 0 || out_w <= 0) throw std::invalid_argument("out_h/out_w must be > 0");
    if (img.ndim() != 2 && img.ndim() != 3) {
        throw std::invalid_argument("img must be 2D (H,W) or 3D (H,W,C)/(C,H,W) float32");
    }

    std::cout << "[npp_wrapper] resize_sqr_pixel_32f start: in_ndim=" << img.ndim()
              << ", out_h=" << out_h << ", out_w=" << out_w
              << ", inter='" << inter_str << "', layout='" << layout << "'"
              << ", shift_y=" << shift_y << ", shift_x=" << shift_x << '\n';

    const auto info = img.request();
    const float* h_in = static_cast<const float*>(info.ptr);

    int in_h = 0;
    int in_w = 0;
    int channels = 1;
    Layout actual_layout = determine_layout(info, layout, in_h, in_w, channels);

    if (channels <= 0) throw std::invalid_argument("channel dimension must be > 0");

    const size_t total_elems = static_cast<size_t>(info.size);

    std::cout << "[npp_wrapper] detected layout=" << layout_to_string(actual_layout)
              << ", in_h=" << in_h << ", in_w=" << in_w
              << ", channels=" << channels << '\n';

    const double scale_y = scale_y_opt.value_or(static_cast<double>(out_h) / static_cast<double>(in_h));
    const double scale_x = scale_x_opt.value_or(static_cast<double>(out_w) / static_cast<double>(in_w));

    std::cout << "[npp_wrapper] scale_y=" << scale_y << ", scale_x=" << scale_x << '\n';

    if (actual_layout == Layout::HWC) {
        if (!(channels == 1 || channels == 3 || channels == 4)) {
            throw std::invalid_argument("Only 1, 3, or 4 interleaved channels are supported; specify layout='chw' for planar data");
        }
    }

    const size_t bytes_in = total_elems * sizeof(float);

    float *d_in = nullptr, *d_out = nullptr;
    size_t bytes_out = static_cast<size_t>(channels) * static_cast<size_t>(out_h) * static_cast<size_t>(out_w) * sizeof(float);

    std::cout << "[npp_wrapper] cudaMalloc d_in=" << bytes_in
              << " bytes, d_out=" << bytes_out << " bytes\n";

    cuda_check(cudaMalloc(&d_in,  bytes_in),  "cudaMalloc d_in");
    cuda_check(cudaMalloc(&d_out, bytes_out), "cudaMalloc d_out");

    std::cout << "[npp_wrapper] cudaMemcpy H2D\n";
    cuda_check(cudaMemcpy(d_in, h_in, bytes_in, cudaMemcpyHostToDevice), "cudaMemcpy H2D");

    py::array_t<float> out;
    if (actual_layout == Layout::SINGLE) {
        out = py::array_t<float>({out_h, out_w});
    } else if (actual_layout == Layout::HWC) {
        out = py::array_t<float>({out_h, out_w, channels});
    } else {
        out = py::array_t<float>({channels, out_h, out_w});
    }
    auto out_info = out.request();
    float* h_out = static_cast<float*>(out_info.ptr);

    NppiSize srcSize { in_w,  in_h  };
    NppiRect srcROI  { 0, 0, in_w,  in_h  };
    NppiSize dstSize { out_w, out_h };
    NppiRect dstROI  { 0, 0, out_w, out_h };

    const auto inter_mode = to_nppi_inter(inter_str);

    auto call_resize_c1 = [&](const float* src_ptr, int src_step, float* dst_ptr, int dst_step) {
        npp_check(nppiResizeSqrPixel_32f_C1R(
            src_ptr, srcSize, src_step, srcROI,
            dst_ptr, dst_step, dstROI,
            scale_x, scale_y, shift_x, shift_y,
            inter_mode
        ), "nppiResizeSqrPixel_32f_C1R");
        std::cout << "[npp_wrapper] nppiResizeSqrPixel_32f_C1R call completed (step=" << src_step
                  << ", scale_x=" << scale_x << ", scale_y=" << scale_y
                  << ", shift_x=" << shift_x << ", shift_y=" << shift_y << ")\n";
    };

    if (actual_layout == Layout::SINGLE) {
        const int in_step = in_w * static_cast<int>(sizeof(float));
        const int out_step = out_w * static_cast<int>(sizeof(float));
        call_resize_c1(d_in, in_step, d_out, out_step);
    } else if (actual_layout == Layout::CHW) {
        const int in_step = in_w * static_cast<int>(sizeof(float));
        const int out_step = out_w * static_cast<int>(sizeof(float));
        const size_t in_plane = static_cast<size_t>(in_h) * static_cast<size_t>(in_w);
        const size_t out_plane = static_cast<size_t>(out_h) * static_cast<size_t>(out_w);
        for (int c = 0; c < channels; ++c) {
            std::cout << "[npp_wrapper] resizing channel " << c << " (CHW)\n";
            const float* src_plane = d_in + c * in_plane;
            float* dst_plane = d_out + c * out_plane;
            call_resize_c1(src_plane, in_step, dst_plane, out_step);
        }
    } else {
        const int in_step = in_w * channels * static_cast<int>(sizeof(float));
        const int out_step = out_w * channels * static_cast<int>(sizeof(float));
        switch (channels) {
            case 1:
                call_resize_c1(d_in, in_step, d_out, out_step);
                break;
            case 3:
                std::cout << "[npp_wrapper] nppiResizeSqrPixel_32f_C3R\n";
                npp_check(nppiResizeSqrPixel_32f_C3R(
                    d_in, srcSize, in_step, srcROI,
                    d_out, out_step, dstROI,
                    scale_x, scale_y, shift_x, shift_y,
                    inter_mode
                ), "nppiResizeSqrPixel_32f_C3R");
                break;
            case 4:
                std::cout << "[npp_wrapper] nppiResizeSqrPixel_32f_C4R\n";
                npp_check(nppiResizeSqrPixel_32f_C4R(
                    d_in, srcSize, in_step, srcROI,
                    d_out, out_step, dstROI,
                    scale_x, scale_y, shift_x, shift_y,
                    inter_mode
                ), "nppiResizeSqrPixel_32f_C4R");
                break;
        }
    }

    std::cout << "[npp_wrapper] cudaMemcpy D2H\n";
    cuda_check(cudaMemcpy(h_out, d_out, bytes_out, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");

    cudaFree(d_in);
    cudaFree(d_out);

    std::cout << "[npp_wrapper] resize_sqr_pixel_32f done\n";

    return out;
}

PYBIND11_MODULE(npp_wrapper, m){
    m.doc() = "Minimal NPP resize wrapper (float32, planar/interleaved)";
    m.def("resize_32f", &resize_32f,
          py::arg("img"), py::arg("out_h"), py::arg("out_w"),
          py::arg("inter") = "bilinear",
          py::arg("layout") = "auto",
          "Resize float32 image using NPP on GPU. Supports HxW, HxWxC, and CxHxW inputs.");
    m.def("resize_sqr_pixel_32f", &resize_sqr_pixel_32f,
          py::arg("img"), py::arg("out_h"), py::arg("out_w"),
          py::arg("scale_y") = py::none(), py::arg("scale_x") = py::none(),
          py::arg("shift_y") = 0.0, py::arg("shift_x") = 0.0,
          py::arg("inter") = "bilinear",
          py::arg("layout") = "auto",
          "Resize float32 image using NPP ResizeSqrPixel (square pixel) API. "
          "Supports HxW, HxWxC, and CxHxW inputs with optional scaling and shifts.");
}
