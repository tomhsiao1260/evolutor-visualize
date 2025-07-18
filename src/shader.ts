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

            uniform sampler2D voldata;
            uniform sampler2D xdata;
            uniform sampler2D ydata;

            void main() {
                float aspect = screenAspect / volumeAspect;
                vec2 uv = vec2((vUv.x - 0.5), (vUv.y - 0.5) / aspect) + vec2(0.5);
                if ( uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 ) return;

                float intensity = texture2D(voldata, uv).r;

                uv = vec2(1.0 - uv.y, uv.x);
                float xValue = texture2D(xdata, uv).r;
                float yValue = texture2D(ydata, uv).r;

                float opacity = 1.0;

                if (yValue > 0.05 && yValue < 0.06) opacity = 0.0;
                if (yValue > 0.15 && yValue < 0.16) opacity = 0.0;
                if (yValue > 0.25 && yValue < 0.26) opacity = 0.0;
                if (yValue > 0.35 && yValue < 0.36) opacity = 0.0;
                if (yValue > 0.45 && yValue < 0.46) opacity = 0.0;
                if (yValue > 0.55 && yValue < 0.56) opacity = 0.0;
                if (yValue > 0.65 && yValue < 0.66) opacity = 0.0;
                if (yValue > 0.75 && yValue < 0.76) opacity = 0.0;
                if (yValue > 0.85 && yValue < 0.86) opacity = 0.0;
                if (yValue > 0.95 && yValue < 0.96) opacity = 0.0;

                if (xValue > 0.05 && xValue < 0.06) opacity = 0.0;
                if (xValue > 0.15 && xValue < 0.16) opacity = 0.0;
                if (xValue > 0.25 && xValue < 0.26) opacity = 0.0;
                if (xValue > 0.35 && xValue < 0.36) opacity = 0.0;
                if (xValue > 0.45 && xValue < 0.46) opacity = 0.0;
                if (xValue > 0.55 && xValue < 0.56) opacity = 0.0;
                if (xValue > 0.65 && xValue < 0.66) opacity = 0.0;
                if (xValue > 0.75 && xValue < 0.76) opacity = 0.0;
                if (xValue > 0.85 && xValue < 0.86) opacity = 0.0;
                if (xValue > 0.95 && xValue < 0.96) opacity = 0.0;

                gl_FragColor = vec4(vec3(intensity), opacity);
            }`,
    });

    this.setValues(params);
  }
}