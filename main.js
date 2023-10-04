import { default as seagulls } from './seagulls.js'
import {Pane} from "https://cdn.jsdelivr.net/npm/tweakpane@4.0.1/dist/tweakpane.min.js";

const sg = await seagulls.init(),
	frag = await seagulls.import("fragment.wgsl"),
	compute_shader = await seagulls.import("compute.wgsl"),
	render_shader = seagulls.constants.vertex + frag,
	size = window.innerWidth * window.innerHeight,
	stateA = new Float32Array(size),
	stateB = new Float32Array(size)


function getRelMid(x, y) {
	const w = Math.floor(window.innerWidth) + x;
	const h = Math.floor(window.innerHeight / 2) + y;
	return h * window.innerWidth + w
}

function setB(radius) {
	for (let i = -1 * radius; i < radius; i++) {
		for (let j = -1 * radius; j < radius; j++) {
			stateB[getRelMid(i, j)] = 1;
		}
	}
}

// set initial state
for (let i = 0; i < size; i++) {
	stateA[i] = 1;
}

setB(1);

// our workgroups are 8x8 in size. To determine
// the number of times we need to run our compute
// shader, we divide our width and height by 8
// and round up.
const workgroup_count = [
	Math.round(window.innerWidth / 8),
	Math.round(window.innerHeight / 8),
	1
]

// Tweakpane setup
const pane = new Pane();

const PARAMS = {
  speed: 1,
  diffusion_a: 1.0,
  diffusion_b: 0.5,
  feed: 0.055,
  kill: 0.061
}

pane.addBinding(
  PARAMS, 
  'speed', 
  {min: 0, max: 2, step:0.1}
);

pane.addBinding(
  PARAMS, 
  'diffusion_a', 
  {min: 0, max: 1, step:0.01}
);

pane.addBinding(
  PARAMS, 
  'diffusion_b', 
  {min: 0.1, max: 1, step:0.01}
);

pane.addBinding(
  PARAMS, 
  'feed', 
  {min: 0, max: 0.1, step:0.001}
);

pane.addBinding(
	PARAMS, 
	'kill', 
	{min: 0, max: 0.1, step:0.001}
  );

// we'll pass two buffers here and initialize them
// both with the state array above. on the first frame,
// the compute shader will read from A and then
// write B... on each subsequent frame this behavior
// will flip. This ensures we're never trying to write
// to the same data that we're reading from, which would
// mess up the game of life simulation (and many others).
sg.buffers({ stateA1: stateA, stateA2: stateA, stateB1: stateB, stateB2: stateB })
	.uniforms({ 
		resolution: [window.innerWidth, window.innerHeight],
		speed: 1,
		diffusion_a: 1.0,
		diffusion_b: 0.5,
		feed: 0.055,
		kill: 0.062 
	})
	.onframe( () => {
		sg.uniforms.speed = PARAMS.speed;
		sg.uniforms.diffusion_a = PARAMS.diffusion_a;
		sg.uniforms.diffusion_b = PARAMS.diffusion_b;
		sg.uniforms.feed = PARAMS.feed;
		sg.uniforms.kill = PARAMS.kill;
	})
	.backbuffer(false)
	// a value of 1 for pingpong() does nothing, but if you
	// turn it up the compute shader will be executed multiple
	// times per frame... you can use this to accelerate the
	// simulation (important for simulations like reaction diffusion).
	.pingpong(50)
	// compute() accepts three arguments:
	// 1) the compute shader itself
	// 2) the number of workgroups to run across three dimensions
	// 3) a list of named buffers to "pingpong", where read/write
	//    operations will flip from one execution to the next with another
	//    paired buffer. in this case, stateA will flip with stateB.
	.compute(
		compute_shader,
		workgroup_count,
		{ pingpong: ['stateA1', 'stateB1']}
	)
	.render(render_shader)
	.run()