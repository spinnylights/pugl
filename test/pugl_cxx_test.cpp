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
   @file pugl_test.c A simple Pugl test that creates a top-level window.
*/

#define GL_SILENCE_DEPRECATION 1

#include "test_utils.h"

#include "pugl/gl.h"
#include "pugl/pugl.hpp"
#include "pugl/pugl_gl.h"

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <utility>

struct CubeData
{
	float    xAngle{0.0};
	float    yAngle{0.0};
	double   lastDrawTime{0.0};
	unsigned framesDrawn{0};
	bool     mouseEntered{false};
	bool     quit{false};
};

using CubeView = pugl::View<CubeData>;

static pugl::Status
onConfigure(CubeView&, const pugl::ConfigureEvent& event)
{
	const auto aspect = static_cast<float>(event.width / event.height);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glClearColor(0.2f, 0.2f, 0.2f, 1.0f);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glViewport(0, 0, int(event.width), int(event.height));

	float proj[16];
	perspective(proj, 1.8f, aspect, 1.0, 100.0f);
	glLoadMatrixf(proj);

	return pugl::Status::success;
}

static pugl::Status
onExpose(CubeView& view, const pugl::ExposeEvent&)
{
	const pugl::World& world    = view.getWorld();
	CubeData&          data     = view.getData();
	const double       thisTime = world.getTime();
	const double       dTime    = thisTime - data.lastDrawTime;
	const double       dAngle   = dTime * 100.0;

	// Rotate
	data.xAngle = fmodf(static_cast<float>(data.xAngle + dAngle), 360.0f);
	data.yAngle = fmodf(static_cast<float>(data.yAngle + dAngle), 360.0f);

	// Set up model matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0f, 0.0f, -10.0f);
	glRotatef(float(data.xAngle), 0.0f, 1.0f, 0.0f);
	glRotatef(float(data.yAngle), 1.0f, 0.0f, 0.0f);

	// Clear background
	const float bg = data.mouseEntered ? 0.2f : 0.1f;
	glClearColor(bg, bg, bg, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Draw cube surfaces
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, cubeStripVertices);
	glColorPointer(3, GL_FLOAT, 0, cubeStripVertices);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 14);
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);

	// Update counters
	data.lastDrawTime = thisTime;
	++data.framesDrawn;

	return pugl::Status::success;
}

static pugl::Status
onKeyPress(CubeView& view, const pugl::KeyPressEvent& event)
{
	if (event.key == PUGL_KEY_ESCAPE || event.key == 'q') {
		view.getData().quit = true;
	}

	return pugl::Status::success;
}

int
main(int argc, char** argv)
{
	const PuglTestOptions opts = puglParseTestOptions(&argc, &argv);

	pugl::World    world;
	CubeView       view{world};
	PuglFpsPrinter fpsPrinter{};

	world.setClassName("Pugl C++ Test");

	view.setFrame({0, 0, 512, 512});
	view.setMinSize(64, 64);
	view.setAspectRatio(1, 1, 16, 9);
	view.setBackend(puglGlBackend());
	view.setHint(PUGL_RESIZABLE, opts.resizable);
	view.setHint(PUGL_SAMPLES, opts.samples);
	view.setHint(PUGL_DOUBLE_BUFFER, opts.doubleBuffer);
	view.setHint(PUGL_SWAP_INTERVAL, opts.doubleBuffer);
	view.setHint(PUGL_IGNORE_KEY_REPEAT, opts.ignoreKeyRepeat);

	view.setEventFunc(onConfigure);
	view.setEventFunc(onExpose);
	view.setEventFunc(onKeyPress);

	view.createWindow("Pugl C++ Test");
	view.showWindow();

	while (!view.getData().quit) {
		view.postRedisplay();
		world.dispatchEvents();

		puglPrintFps(world.cobj(), &fpsPrinter, &view.getData().framesDrawn);
	}

	return 0;
}
