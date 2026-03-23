// ============================================================================
// Qwen3-TTS HTTP Server (Simplified & Fixed)
// Based on predict-woo/qwen3-tts.cpp  
// ============================================================================

#define _USE_MATH_DEFINES
#include "qwen3_tts.h"
#include "httplib.h"
#include "json.hpp"

#include <iostream>
#include <thread>

using json = nlohmann::json;

// ============================================================================
// Base64 encoding
// ============================================================================

static std::string base64_encode(const uint8_t* data, size_t len) {
    static const char b64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result;
    int i = 0;
    uint32_t triple;
    
    while (i < len) {
        triple = (i < len ? data[i++] : 0) << 16;
        if (i < len) triple |= data[i++] << 8;
        if (i < len) triple |= data[i++];
        
        result += b64[(triple >> 18) & 0x3F];
        result += b64[(triple >> 12) & 0x3F];
        result += b64[(triple >> 6) & 0x3F];
        result += b64[triple & 0x3F];
    }
    
    while (result.size() % 4) result += '=';
    return result;
}

// ============================================================================
// WAV encoding  
// ============================================================================

static std::vector<uint8_t> float_to_wav(const std::vector<float>& samples, int sample_rate) {
    std::vector<int16_t> pcm(samples.size());
    for (size_t i = 0; i < samples.size(); i++) {
        float s = samples[i] * 32767.0f;
        pcm[i] = (s > 32767) ? 32767 : ((s < -32768) ? -32768 : (int16_t)s);
    }
    
    std::vector<uint8_t> wav(44 + pcm.size() * 2);
    
    // RIFF header
    const char* riff = "RIFF";
    memcpy(wav.data(), riff, 4);
    uint32_t file_size = 36 + (uint32_t)(pcm.size() * 2);
    memcpy(wav.data() + 4, &file_size, 4);
    
    const char* wave = "WAVE";
    memcpy(wav.data() + 8, wave, 4);
    
    // fmt chunk
    const char* fmt = "fmt ";
    memcpy(wav.data() + 12, fmt, 4);
    uint32_t fmt_size = 16;
    memcpy(wav.data() + 16, &fmt_size, 4);
    
    uint16_t audio_format = 1;  // PCM
    memcpy(wav.data() + 20, &audio_format, 2);
    uint16_t num_channels = 1;
    memcpy(wav.data() + 22, &num_channels, 2);
    memcpy(wav.data() + 24, &sample_rate, 4);
    
    uint32_t byte_rate = sample_rate * 2;
    memcpy(wav.data() + 28, &byte_rate, 4);
    
    uint16_t block_align = 2;
    memcpy(wav.data() + 32, &block_align, 2);
    uint16_t bits_per_sample = 16;
    memcpy(wav.data() + 34, &bits_per_sample, 2);
    
    // data chunk
    const char* data_str = "data";
    memcpy(wav.data() + 36, data_str, 4);
    uint32_t data_size = (uint32_t)(pcm.size() * 2);
    memcpy(wav.data() + 40, &data_size, 4);
    
    // PCM data
    memcpy(wav.data() + 44, pcm.data(), pcm.size() * 2);
    
    return wav;
}

// ============================================================================
// HTTP Server
// ============================================================================

int start_http_server(const std::string& model_dir, qwen3_tts::tts_params base_params, 
                      int port = 5002) {
    
    fprintf(stderr, "Qwen3-TTS HTTP Server\n");
    fprintf(stderr, "Loading models from: %s\n", model_dir.c_str());
    
    qwen3_tts::Qwen3TTS tts;
    if (!tts.load_models(model_dir)) {
        fprintf(stderr, "Error: %s\n", tts.get_error().c_str());
        return 1;
    }
    
    fprintf(stderr, "Models loaded!\n");
    fprintf(stderr, "Starting server on port %d...\n\n", port);
    
    httplib::Server svr;
    
    // Health check
    svr.Post("/health", [](const httplib::Request&, httplib::Response& res) {
        nlohmann::json j;
        j["status"] = "ok";
        j["service"] = "qwen3-tts-server";
        res.set_content(j.dump(), "application/json");
    });
    
    // Basic synthesis
    svr.Post("/generate", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            nlohmann::json j = nlohmann::json::parse(req.body);
            
            std::string text = j.value("text", "");
            if (text.empty()) {
                nlohmann::json err;
                err["error"] = "Missing 'text' parameter";
                res.set_content(err.dump(), "application/json");
                return;
            }
            
            // Apply params
            qwen3_tts::tts_params params = base_params;
            if (j.contains("temperature")) params.temperature = j["temperature"].get<float>();
            if (j.contains("top_k")) params.top_k = j["top_k"].get<int>();
            if (j.contains("language_id")) params.language_id = j["language_id"].get<int>();
            if (j.contains("language")) {
                std::string lang = j["language"];
                if (lang == "en") params.language_id = 2050;
                else if (lang == "es") params.language_id = 2054;
                else if (lang == "zh") params.language_id = 2055;
                else if (lang == "ja") params.language_id = 2058;
                else if (lang == "ko") params.language_id = 2064;
                else if (lang == "ru") params.language_id = 2069;
                else if (lang == "de") params.language_id = 2053;
                else if (lang == "fr") params.language_id = 2061;
            }
            if (j.contains("print_timing")) params.print_timing = j["print_timing"].get<bool>();
            
            auto result = tts.synthesize(text, params);
            
            if (!result.success) {
                nlohmann::json err;
                err["error"] = result.error_msg;
                res.set_content(err.dump(), "application/json");
                return;
            }
            
            auto wav = float_to_wav(result.audio, result.sample_rate);
            res.set_content(reinterpret_cast<const char*>(wav.data()), wav.size(), "audio/wav");
            
        } catch (const std::exception& e) {
            nlohmann::json err;
            err["error"] = std::string("JSON error: ") + e.what();
            res.set_content(err.dump(), "application/json");
        }
    });
    
    // Voice cloning
    svr.Post("/clone", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            nlohmann::json j = nlohmann::json::parse(req.body);
            
            std::string text = j.value("text", "");
            if (text.empty()) {
                nlohmann::json err;
                err["error"] = "Missing 'text' parameter";
                res.set_content(err.dump(), "application/json");
                return;
            }
            
            std::string reference = j.value("reference", "");
            if (reference.empty()) {
                nlohmann::json err;
                err["error"] = "Missing 'reference' parameter (audio file path)";
                res.set_content(err.dump(), "application/json");
                return;
            }
            
            qwen3_tts::tts_params params = base_params;
            auto result = tts.synthesize_with_voice(text, reference, params);
            
            if (!result.success) {
                nlohmann::json err;
                err["error"] = result.error_msg;
                res.set_content(err.dump(), "application/json");
                return;
            }
            
            auto wav = float_to_wav(result.audio, result.sample_rate);
            res.set_content(reinterpret_cast<const char*>(wav.data()), wav.size(), "audio/wav");
            
        } catch (const std::exception& e) {
            nlohmann::json err;
            err["error"] = std::string("Error: ") + e.what();
            res.set_content(err.dump(), "application/json");
        }
    });
    
    // Note: /extract-spk endpoint removed - requires audio decoding not available
    // Use CLI tool (qwen3-tts-cli) for embedding extraction instead
    
    svr.listen("0.0.0.0", port);
    return 0;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    std::string model_dir;
    int port = 5002;
    
    qwen3_tts::tts_params params;
    params.temperature = 0.9f;
    params.top_k = 50;
    params.language_id = 2050;
    params.n_threads = 4;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-m" || arg == "--model") {
            if (++i >= argc) return 1;
            model_dir = argv[i];
        } else if (arg == "--port") {
            if (++i >= argc) return 1;
            port = std::stoi(argv[i]);
        } else if (arg == "--temperature") {
            if (++i >= argc) return 1;
            params.temperature = std::stof(argv[i]);
        } else if (arg == "-h" || arg == "--help") {
            fprintf(stderr, "Usage: %s -m <model_dir> [--port 5002]\n", argv[0]);
            return 0;
        }
    }
    
    if (model_dir.empty()) {
        fprintf(stderr, "Error: model directory (-m) is required\n");
        return 1;
    }
    
    return start_http_server(model_dir, params, port);
}