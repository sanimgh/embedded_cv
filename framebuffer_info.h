// fichier framebuffer_info.h
#ifndef FRAMEBUFFER_INFO_H
#define FRAMEBUFFER_INFO_H

#include <stdint.h>

struct framebuffer_info {
    uint32_t bits_per_pixel;
    uint32_t xres_virtual;
    uint32_t xres;
};

struct framebuffer_info get_framebuffer_info(const char *framebuffer_device_path);

#endif // FRAMEBUFFER_INFO_H
