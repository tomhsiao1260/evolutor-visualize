import './style.css'
import * as THREE from 'three'
import { MOUSE, TOUCH } from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { NRRDLoader } from 'three/examples/jsm/loaders/NRRDLoader';
import { TIFFLoader } from 'three/addons/loaders/TIFFLoader.js'
import { GUI } from 'three/examples/jsm/libs/lil-gui.module.min'
import { Shader } from './shader.ts'
import { slice, openArray } from "zarr"
import textureTab20 from './textures/tab20.png'

window.addEventListener('resize', () =>
{
    // Update camera
    camera.aspect = window.innerWidth / window.innerHeight
    camera.updateProjectionMatrix()

    // Update renderer
    renderer.setSize(window.innerWidth, window.innerHeight)
    render()
})

// Scene
const scene = new THREE.Scene()

// Camera
const aspect = window.innerWidth / window.innerHeight
const camera = new THREE.OrthographicCamera(-1 * aspect, 1 * aspect, 1, -1, 0.01, 100)
camera.up.set(0, -1, 0)
camera.position.z = -1.3
scene.add(camera)


// Mesh
const material = new Shader()
// const material = new THREE.MeshNormalMaterial()
const geometry = new THREE.PlaneGeometry(2, 2, 100, 100)
const mesh = new THREE.Mesh(geometry, material)
scene.add(mesh)

// renderer setup
const canvas = document.querySelector('.webgl')
const renderer = new THREE.WebGLRenderer({ antialias: true, canvas: canvas })
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
renderer.setSize(window.innerWidth, window.innerHeight)
renderer.setClearColor(0, 0)
renderer.outputColorSpace = THREE.SRGBColorSpace

// Controls
const controls = new OrbitControls(camera, canvas)
controls.enableDamping = false
controls.screenSpacePanning = true // pan orthogonal to world-space direction camera.up
controls.mouseButtons = { LEFT: MOUSE.PAN, MIDDLE: MOUSE.DOLLY, RIGHT: MOUSE.ROTATE }
controls.touches = { ONE: TOUCH.PAN, TWO: TOUCH.DOLLY_PAN }
controls.addEventListener('change', render)

// Tick
function render() {
    renderer.render(scene, camera)
}
render()

// GUI
const gui = new GUI() 
const params = { flatten: 0, opacity: 0.5, axis: 0 }

gui.add(params, 'flatten', 0, 1, 0.01).onChange(() => {
  material.uniforms.flatten.value = params.flatten
  render()
})
gui.add(params, 'opacity', 0, 1, 0.01).onChange(() => {
  material.uniforms.opacity.value = params.opacity
  render()
})
gui.add(params, 'axis', 0, 1, 0.01).onChange(() => {
  material.uniforms.axis.value = params.axis
  render()
})

// Data
const nrrdLoader = new NRRDLoader()
const tiffLoader = new TIFFLoader()

loadData()

async function loadData() {
  const level = 2
  const chunk = 128 * 2
  const x0 = 896 + 0
  const y0 = 896 + 0

  const volZarr = await openArray({
    store: "http://localhost:5173/",
    path: `scroll.zarr/${level}`,
    mode: "r"
  })

  const { data: volData, shape } = await volZarr.get([0, slice(y0, y0+chunk), slice(x0, x0+chunk)])
  const [h, w] = shape
  const volume = new Uint8ClampedArray(w * h)
  for (let i = 0; i < h; ++i) {
    for (let j = 0; j < w; ++j) {
      volume[chunk*i+j] = volData[i][j] / 256
    }
  }

  const volumeTex = new THREE.DataTexture(volume, h, w)
  volumeTex.format = THREE.RedFormat
  volumeTex.type = THREE.UnsignedByteType
  volumeTex.minFilter = THREE.LinearFilter
  volumeTex.magFilter = THREE.NearestFilter
  volumeTex.needsUpdate = true

  material.uniforms.volumeAspect.value = w / h
  material.uniforms.screenAspect.value = 2 / 2
  material.uniforms.voldata.value = volumeTex

  const uZarr = await openArray({
    store: "http://localhost:5173/",
    path: `scroll_u.zarr/${level}`,
    mode: "r"
  })
  const vZarr = await openArray({
    store: "http://localhost:5173/",
    path: `scroll_v.zarr/${level}`,
    mode: "r"
  })

  const { data: uData } = await uZarr.get([0, slice(y0, y0+chunk), slice(x0, x0+chunk)])
  const u_coord = new Uint8ClampedArray(w * h)
  for (let i = 0; i < h; ++i) {
    for (let j = 0; j < w; ++j) {
      u_coord[chunk*i+j] = uData[i][j] / 256
    }
  }
  const { data: vData } = await vZarr.get([0, slice(y0, y0+chunk), slice(x0, x0+chunk)])
  const v_coord = new Uint8ClampedArray(w * h)
  for (let i = 0; i < h; ++i) {
    for (let j = 0; j < w; ++j) {
      v_coord[chunk*i+j] = vData[i][j] / 256
    }
  }

  const uTex = new THREE.DataTexture(u_coord, h, w)
  uTex.format = THREE.RedFormat
  uTex.type = THREE.UnsignedByteType
  uTex.minFilter = THREE.LinearFilter
  uTex.magFilter = THREE.NearestFilter
  uTex.needsUpdate = true

  const vTex = new THREE.DataTexture(v_coord, h, w)
  vTex.format = THREE.RedFormat
  vTex.type = THREE.UnsignedByteType
  vTex.minFilter = THREE.LinearFilter
  vTex.magFilter = THREE.NearestFilter
  vTex.needsUpdate = true

  material.uniforms.udata.value = uTex
  material.uniforms.vdata.value = vTex

  const tab20 = new THREE.TextureLoader().load(textureTab20)
  tab20.minFilter = THREE.NearestFilter
  tab20.maxFilter = THREE.NearestFilter
  material.uniforms.tab20.value = tab20

  render()
}



