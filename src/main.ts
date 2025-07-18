import './style.css'
import * as THREE from 'three'
import { MOUSE, TOUCH } from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { NRRDLoader } from 'three/examples/jsm/loaders/NRRDLoader';
import { TIFFLoader } from 'three/addons/loaders/TIFFLoader.js'
import { Shader } from './shader.ts'

window.addEventListener('resize', () =>
{
    // Update camera
    camera.aspect = window.innerWidth / window.innerHeight
    camera.updateProjectionMatrix()

    // Update renderer
    renderer.setSize(window.innerWidth, window.innerHeight)
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

// Data
const nrrdLoader = new NRRDLoader()
const tiffLoader = new TIFFLoader()

Promise.all([
  tiffLoader.loadAsync('volume.tif'),
  nrrdLoader.loadAsync('x.nrrd'),
  nrrdLoader.loadAsync('y.nrrd')
]).then(([volumeTex, xgrid, ygrid]) => {
    const { width, height } = volumeTex.source.data

    volumeTex.magFilter = THREE.NearestFilter
    volumeTex.minFilter = THREE.LinearFilter

    material.uniforms.volumeAspect.value = width / height
    material.uniforms.screenAspect.value = 2 / 2
    material.uniforms.voldata.value = volumeTex
    console.log(volumeTex, xgrid, ygrid)

    render()
})

