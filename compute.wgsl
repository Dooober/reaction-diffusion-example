@group(0) @binding(0) var<uniform> res: vec2f;
@group(0) @binding(1) var<uniform> speed: f32;
@group(0) @binding(2) var<uniform> diffusion_a: f32;
@group(0) @binding(3) var<uniform> diffusion_b: f32;
@group(0) @binding(4) var<uniform> feed: f32;
@group(0) @binding(5) var<uniform> kill: f32;
@group(0) @binding(6) var<storage, read_write> stateA1: array<f32>;
@group(0) @binding(7) var<storage, read_write> stateA2: array<f32>;
@group(0) @binding(8) var<storage, read_write> stateB1: array<f32>;
@group(0) @binding(9) var<storage, read_write> stateB2: array<f32>;


// Get index of (x, y) position
fn index( x:u32, y:u32 ) -> u32 {
	return y * u32(res.x) + x;
}

// Laplace with given weights on A values
fn laplaceA(i:u32, cell:vec3u) -> f32 {
	let sum = 	-1.0 * stateA1[i] +
				0.05 * stateA1[ index(cell.x + 1u, cell.y + 1u) ] + // upper right
				0.20 * stateA1[ index(cell.x + 1u, cell.y)      ] + // right
				0.05 * stateA1[ index(cell.x + 1u, cell.y - 1u) ] + // lower right
				0.20 * stateA1[ index(cell.x, cell.y - 1u)      ] + // down 
				0.05 * stateA1[ index(cell.x - 1u, cell.y - 1u) ] + // lower left
				0.20 * stateA1[ index(cell.x - 1u, cell.y)      ] + // left
				0.05 * stateA1[ index(cell.x - 1u, cell.y + 1u) ] + // upper left
				0.20 * stateA1[ index(cell.x, cell.y + 1u)      ];  // up
	return sum;
}

fn laplaceB(i:u32, cell:vec3u) -> f32 {
	let sum = 	-1.0 * stateB1[i] +
				0.05 * stateB1[ index(cell.x + 1u, cell.y + 1u) ] + // upper right
				0.20 * stateB1[ index(cell.x + 1u, cell.y)      ] + // right
				0.05 * stateB1[ index(cell.x + 1u, cell.y - 1u) ] + // lower right
				0.20 * stateB1[ index(cell.x, cell.y - 1u)      ] + // down 
				0.05 * stateB1[ index(cell.x - 1u, cell.y - 1u) ] + // lower left
				0.20 * stateB1[ index(cell.x - 1u, cell.y)      ] + // left
				0.05 * stateB1[ index(cell.x - 1u, cell.y + 1u) ] + // upper left
				0.20 * stateB1[ index(cell.x, cell.y + 1u)      ];  // up
	return sum;
}

// This allows the implementation of orientation
fn timeChange(x:u32, t:f32) -> f32 {
	return (f32(x) / res.x) * t * 0.5;
}

// here we specify a workgroup size of 64. this
// should always be a power of 8 for optimization on
// nvidia cards, 64 is a common default size that
// should work on most graphics cards.
@compute @workgroup_size(8,8)

fn cs(
	@builtin(global_invocation_id) cell:vec3u
)  {
	let dt = timeChange(cell.x, speed);

	// get the index of this particular cell
	let i = index(cell.x, cell.y);

	let a = stateA1[i];
	let b = stateB1[i];
	stateA2[i] = a + (diffusion_a * laplaceA(i, cell) - a*b*b + feed*(1-a)) * dt;
	stateB2[i] = b + (diffusion_b * laplaceB(i, cell) + a*b*b - (kill+feed)*b) * dt;
}