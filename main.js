import { default as seagulls } from './seagulls.js'
import { default as Video    } from './video.js'
import { default as Audio    } from './audio.js'

const shader = `
@group(0) @binding(0) var<uniform> frame: f32;
@group(0) @binding(1) var<uniform> res:   vec2f;
@group(0) @binding(2) var<uniform> audio: vec3f;
@group(0) @binding(3) var<uniform> mouse: vec3f;
@group(0) @binding(4) var backSampler:    sampler;
@group(0) @binding(5) var backBuffer:     texture_2d<f32>;
@group(0) @binding(6) var videoSampler:   sampler;
@group(1) @binding(0) var videoBuffer:    texture_external;

@vertex 
fn vs( @location(0) input : vec2f ) ->  @builtin(position) vec4f {
  return vec4f( input, 0., 1.); 
}

@fragment 
fn fs( @builtin(position) pos : vec4f ) -> @location(0) vec4f {
  let vid = textureSampleBaseClampToEdge( videoBuffer, videoSampler, pos.xy / res );
  let fb  = textureSample( backBuffer, backSampler, pos.xy / res );
  let out = vid * .05 + fb * .95;
  return vec4f( out.r, 0.,0. , 1. );
}`

async function main() {
  let frame = 0

  document.body.onclick = Audio.start

  await Video.init()

  const sg = await seagulls.init()

  sg.uniforms({ 
    frame:0, 
    res:[window.innerWidth, window.innerHeight],
    audio:[0,0,0],
    mouse:[0,0,0],
  })
  .onframe( ()=> {
    sg.uniforms.frame = frame++ 
    sg.uniforms.audio = [ Audio.low, Audio.mid, Audio.high ]
  })
  .textures([ Video.element ]) 
  .render( shader, { uniforms: ['frame','res', 'audio', 'mouse' ] })
  .run()
}

main()
