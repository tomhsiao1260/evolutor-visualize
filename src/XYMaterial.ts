import { ShaderMaterial, BackSide } from 'three';

export class XYMaterial extends ShaderMaterial {
  constructor(params) {
    super({
        transparent: true,
        side: BackSide,

        uniforms: {
            volumeAspect : { value: 100 / 100 },
            screenAspect : { value: 2 / 2 },
            xdata : { value: null },
            ydata : { value: null },
        },

        vertexShader: /* glsl */ `
            varying vec2 vUv;

            void main() {
                vUv = vec2(uv.x, 1.0 - uv.y);
                // gl_Position = vec4(position, 1.0);
                gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
            }`,

        fragmentShader: /* glsl */ `
            varying vec2 vUv;
            uniform float volumeAspect;
            uniform float screenAspect;
            uniform sampler2D xdata;
            uniform sampler2D ydata;

            void main() {
                float aspect = screenAspect / volumeAspect;
                vec2 uv = vec2((vUv.x - 0.5), (vUv.y - 0.5) / aspect) + vec2(0.5);
                if ( uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 ) return;

                uv = vec2(1.0 - uv.y, uv.x);

                float xValue = texture2D(xdata, uv).r;
                float yValue = texture2D(ydata, uv).r;

                if (yValue > 0.6 || yValue < 0.5) return;
                gl_FragColor = vec4(1.0);
            }`,
    });

    this.setValues(params);
  }
}