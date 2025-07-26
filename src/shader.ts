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
            flatten : { value: 0 },
            opacity : { value: 0.5 },
            axis : { value: 0 },
        },

        vertexShader: /* glsl */ `
            varying vec2 vUv;
            uniform sampler2D udata;
            uniform sampler2D vdata;
            uniform float flatten;

            void main() {
                vUv = vec2(uv.x, 1.0 - uv.y);

                float uValue = texture2D(udata, uv).r;
                float vValue = texture2D(vdata, uv).r;

                vec3 pos = position;
                pos.x = (1.0 - flatten) * position.x - flatten * 2.0 * (uValue - 0.5);
                pos.y = (1.0 - flatten) * position.y + flatten * 2.0 * (vValue - 0.5);

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

            void main() {
                float aspect = screenAspect / volumeAspect;
                vec2 uv = vec2((vUv.x - 0.5), (vUv.y - 0.5) / aspect) + vec2(0.5);
                if ( uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 ) return;

                uv.y = 1.0 - uv.y;

                float v = texture2D(voldata, uv).r;

                uv = vec2(1.0 - uv.y, uv.x);
                float uValue = texture2D(udata, uv).r;
                float vValue = texture2D(vdata, uv).r;

                vec3 uColor = vec3(0.0);
                if (uValue < 0.1) {
                    uColor = vec3(31.0, 119.0, 180.0);
                } else if (uValue < 0.2) {
                    uColor = vec3(255.0, 127.0, 14.0);
                } else if (uValue < 0.3) {
                    uColor = vec3(214.0, 39.0, 40.0);
                } else if (uValue < 0.4) {
                    uColor = vec3(44.0, 160.0, 44.0);
                } else if (uValue < 0.5) {
                    uColor = vec3(148.0, 103.0, 189.0);
                } else if (uValue < 0.6) {
                    uColor = vec3(140.0, 86.0, 75.0);
                } else if (uValue < 0.7) {
                    uColor = vec3(227.0, 119.0, 194.0);
                } else if (uValue < 0.8) {
                    uColor = vec3(127.0, 127.0, 127.0);
                } else if (uValue < 0.9) {
                    uColor = vec3(188.0, 189.0, 34.0);
                } else {
                    uColor = vec3(23.0, 190.0, 207.0);
                }
                uColor /= 255.0;

                vec3 vColor = vec3(0.0);
                if (vValue < 0.1) {
                    vColor = vec3(31.0, 119.0, 180.0);
                } else if (vValue < 0.2) {
                    vColor = vec3(255.0, 127.0, 14.0);
                } else if (vValue < 0.3) {
                    vColor = vec3(214.0, 39.0, 40.0);
                } else if (vValue < 0.4) {
                    vColor = vec3(44.0, 160.0, 44.0);
                } else if (vValue < 0.5) {
                    vColor = vec3(148.0, 103.0, 189.0);
                } else if (vValue < 0.6) {
                    vColor = vec3(140.0, 86.0, 75.0);
                } else if (vValue < 0.7) {
                    vColor = vec3(227.0, 119.0, 194.0);
                } else if (vValue < 0.8) {
                    vColor = vec3(127.0, 127.0, 127.0);
                } else if (vValue < 0.9) {
                    vColor = vec3(188.0, 189.0, 34.0);
                } else {
                    vColor = vec3(23.0, 190.0, 207.0);
                }
                vColor /= 255.0;

                vec3 color = uColor * axis + vColor * (1.0 - axis);
                vec3 c = vec3(v) * (1.0 - opacity) + color * opacity;
                gl_FragColor = vec4(c, 1.0);
            }`,
    });

    this.setValues(params);
  }
}