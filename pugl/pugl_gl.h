/*
  Copyright 2012-2019 David Robillard <http://drobilla.net>

  Permission to use, copy, modify, and/or distribute this software for any
  purpose with or without fee is hereby granted, provided that the above
  copyright notice and this permission notice appear in all copies.

  THIS SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
*/

/**
   @file pugl_gl.h OpenGL-specific API.
*/

#ifndef PUGL_PUGL_GL_H
#define PUGL_PUGL_GL_H

#include "pugl/pugl.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
   OpenGL extension function.
*/
typedef void (*PuglGlFunc)(void);

/**
   Return the address of an OpenGL extension function.
*/
PUGL_API PuglGlFunc
puglGetProcAddress(const char* name);

/**
   OpenGL graphics backend.

   Pass the return value to puglInitBackend() to draw to a view with OpenGL.
*/
PUGL_API const PuglBackend*
puglGlBackend(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif // PUGL_PUGL_GL_H
