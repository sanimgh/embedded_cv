// fichier framebuffer_info.c ou framebuffer_info.cpp
#include "framebuffer_info.h"
#include <stdio.h>
#include <fcntl.h>
#include <linux/fb.h>
#include <sys/ioctl.h>

struct framebuffer_info get_framebuffer_info(const char *framebuffer_device_path)
{
    struct framebuffer_info fb_info;        // Used to return the required attrs.
    struct fb_var_screeninfo screen_info;   // Used to get attributes of the device from OS kernel.

	int fd = open(framebuffer_device_path, O_RDWR);

	if(ioctl(fd,FBIOGET_VSCREENINFO,&screen_info)<0)
	{
		printf("ioctl fail\n");	
	}
	fb_info.xres_virtual=screen_info.xres_virtual;
	fb_info.bits_per_pixel=screen_info.bits_per_pixel;
	fb_info.xres=screen_info.xres;


    return fb_info;
};