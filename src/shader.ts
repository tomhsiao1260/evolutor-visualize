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
            xdata : { value: null },
            ydata : { value: null },
            flatten : { value: 0 },
            opacity : { value: 0.5 },
            axis : { value: 0 },
        },

        vertexShader: /* glsl */ `
            varying vec2 vUv;
            uniform sampler2D xdata;
            uniform sampler2D ydata;
            uniform float flatten;

            void main() {
                vUv = vec2(uv.x, 1.0 - uv.y);

                vec2 uv = vec2(1.0 - vUv.y, vUv.x);
                float xValue = texture2D(xdata, uv).r;
                float yValue = texture2D(ydata, uv).r;

                vec3 pos = position;
                pos.x = (1.0 - flatten) * position.x + flatten * 2.0 * (xValue - 0.5);
                pos.y = (1.0 - flatten) * position.y + flatten * 2.0 * (yValue - 0.5);

                gl_Position = projectionMatrix * modelViewMatrix * vec4( pos, 1.0 );
            }`,

        fragmentShader: /* glsl */ `
            varying vec2 vUv;
            uniform float volumeAspect;
            uniform float screenAspect;
            uniform float opacity;
            uniform float axis;

            uniform sampler2D voldata;
            uniform sampler2D xdata;
            uniform sampler2D ydata;

            void main() {
                float aspect = screenAspect / volumeAspect;
                vec2 uv = vec2((vUv.x - 0.5), (vUv.y - 0.5) / aspect) + vec2(0.5);
                if ( uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 ) return;

                float v = texture2D(voldata, uv).r;

                uv = vec2(1.0 - uv.y, uv.x);
                float xValue = texture2D(xdata, uv).r;
                float yValue = texture2D(ydata, uv).r;

                vec3 xColor = vec3(0.0);
                if (xValue < 0.1) {
                    xColor = vec3(31.0, 119.0, 180.0);
                } else if (xValue < 0.2) {
                    xColor = vec3(255.0, 127.0, 14.0);
                } else if (xValue < 0.3) {
                    xColor = vec3(214.0, 39.0, 40.0);
                } else if (xValue < 0.4) {
                    xColor = vec3(44.0, 160.0, 44.0);
                } else if (xValue < 0.5) {
                    xColor = vec3(148.0, 103.0, 189.0);
                } else if (xValue < 0.6) {
                    xColor = vec3(140.0, 86.0, 75.0);
                } else if (xValue < 0.7) {
                    xColor = vec3(227.0, 119.0, 194.0);
                } else if (xValue < 0.8) {
                    xColor = vec3(127.0, 127.0, 127.0);
                } else if (xValue < 0.9) {
                    xColor = vec3(188.0, 189.0, 34.0);
                } else {
                    xColor = vec3(23.0, 190.0, 207.0);
                }
                xColor /= 255.0;

                vec3 yColor = vec3(0.0);
                if (yValue < 0.1) {
                    yColor = vec3(31.0, 119.0, 180.0);
                } else if (yValue < 0.2) {
                    yColor = vec3(255.0, 127.0, 14.0);
                } else if (yValue < 0.3) {
                    yColor = vec3(214.0, 39.0, 40.0);
                } else if (yValue < 0.4) {
                    yColor = vec3(44.0, 160.0, 44.0);
                } else if (yValue < 0.5) {
                    yColor = vec3(148.0, 103.0, 189.0);
                } else if (yValue < 0.6) {
                    yColor = vec3(140.0, 86.0, 75.0);
                } else if (yValue < 0.7) {
                    yColor = vec3(227.0, 119.0, 194.0);
                } else if (yValue < 0.8) {
                    yColor = vec3(127.0, 127.0, 127.0);
                } else if (yValue < 0.9) {
                    yColor = vec3(188.0, 189.0, 34.0);
                } else {
                    yColor = vec3(23.0, 190.0, 207.0);
                }
                yColor /= 255.0;

                vec3 color = xColor * axis + yColor * (1.0 - axis);
                vec3 c = vec3(v) * (1.0 - opacity) + color * opacity;
                gl_FragColor = vec4(c, 1.0);
            }`,
    });

    this.setValues(params);
  }
}