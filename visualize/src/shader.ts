import { ShaderMaterial, BackSide } from 'three';

export class Shader extends ShaderMaterial {
  constructor(params) {
    super({
        transparent: true,
        side: BackSide,

        uniforms: {
            volumeAspect : { value: 100 / 100 },
            screenAspect : { value: 2 / 2 },
            voldata : { value: null },
            udata : { value: null },
            vdata : { value: null },
            tab20 : { value: null },
            flatten : { value: 0 },
            opacity : { value: 0.5 },
            axis : { value: 0 },
            cx : { value: 0 },
            cy : { value: 0 },
        },

        vertexShader: /* glsl */ `
            # define PI 3.141592653589793

            varying vec2 vUv;
            uniform sampler2D udata;
            uniform sampler2D vdata;
            uniform float flatten;
            uniform float cx;
            uniform float cy;

            void main() {
                vUv = vec2(uv.x, 1.0 - uv.y);

                float uValue = texture2D(udata, uv).r;
                float vValue = texture2D(vdata, uv).r;

                vec2 c = vec2(cx, cy);
                float ro = length(1.0 - c);
                float th = 2.0 * PI * uValue;
                vec2 n = vec2(cos(th), sin(th)) * -1.0;
                vec2 p = ro * vValue * n;
                p = (p + c) * 2.0 - 1.0;

                vec3 pos = (position + 1.0) / 2.0;
                pos.x = (1.0 - flatten) * position.x + flatten * p.x;
                pos.y = (1.0 - flatten) * position.y + flatten * p.y;

                gl_Position = projectionMatrix * modelViewMatrix * vec4( pos, 1.0 );
            }`,

        fragmentShader: /* glsl */ `
            varying vec2 vUv;
            uniform float volumeAspect;
            uniform float screenAspect;
            uniform float opacity;
            uniform float axis;

            uniform sampler2D voldata;
            uniform sampler2D udata;
            uniform sampler2D vdata;
            uniform sampler2D tab20;

            void main() {
                float aspect = screenAspect / volumeAspect;
                vec2 uv = vec2((vUv.x - 0.5), (vUv.y - 0.5) / aspect) + vec2(0.5);
                if ( uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 ) return;

                uv.y = 1.0 - uv.y;

                float v = texture2D(voldata, uv).r;

                float uValue = texture2D(udata, uv).r;
                float vValue = texture2D(vdata, uv).r;

                vec3 uColor = texture2D(tab20, vec2(uValue, 0.5)).rgb;
                vec3 vColor = texture2D(tab20, vec2(vValue, 0.5)).rgb;

                vec3 color = uColor * axis + vColor * (1.0 - axis);
                vec3 c = vec3(v) * (1.0 - opacity) + color * opacity;
                gl_FragColor = vec4(c, 1.0);
            }`,
    });

    this.setValues(params);
  }
}