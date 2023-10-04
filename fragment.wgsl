@group(0) @binding(0) var<uniform> res:   vec2f;
@group(0) @binding(1) var<uniform> speed: f32;
@group(0) @binding(2) var<uniform> diffusion_a: f32;
@group(0) @binding(3) var<uniform> diffusion_b: f32;
@group(0) @binding(4) var<uniform> feed: f32;
@group(0) @binding(5) var<uniform> kill: f32;
@group(0) @binding(6) var<storage> stateA: array<f32>;
@group(0) @binding(7) var<storage> stateB: array<f32>;

@fragment 
fn fs( @builtin(position) pos : vec4f ) -> @location(0) vec4f {
    let i : u32 = u32( pos.y * res.x + pos.x );
    let a = stateA[i];
    let b = stateB[i];
    return vec4f( b,0.,a, 1.);
}