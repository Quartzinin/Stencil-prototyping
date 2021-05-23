import "./edgeHandling"

-- code is based on
-- https://gitlab.com/larisa.stoltzfus/liftstencil-cgo2018-artifact/-/blob/master/benchmarks/figure7/workflow1/reference/hotspot3D/hotspotKernel.cl

-- constants
let amb_temp: f32 = 80.0
let max_pd: f32 = 3.0e6
let precision: f32 = 0.001
let spec_heat_si: f32 = 1.75e6
let k_si: f32 = 100.0
let factor_chip: f32 = 0.5
let t_chip: f32 = 0.0005
let chip_height: f32 = 0.016
let chip_width: f32 = 0.016

let calculate_update
        (sdc, cew, cns, ctb, cc)
        C_pow B N W C E S T : f32 =
  cc*C + cew*W + cew*E + cns*S + cns*N + ctb*B + ctb*T + sdc*C_pow + ctb*amb_temp

let single_iteration_maps [Nz][Ny][Nx]
    (temp: [Nz][Ny][Nx]f32) (power: [Nz][Ny][Nx]f32)
    updater
    : [Nz][Ny][Nx]f32 =
  let bound = edgeHandling.extendEdge3D temp (Nz-1) (Ny-1) (Nx-1)
  in tabulate_3d Nz Ny Nx (\z y x ->
        let C_pow = power[z,y,x]
        let rbound = bound z y x
        let B = rbound (-1) ( 0) ( 0)
        let N = rbound ( 0) (-1) ( 0)
        let W = rbound ( 0) ( 0) (-1)
        let C = rbound ( 0) ( 0) ( 0)
        let E = rbound ( 0) ( 0) ( 1)
        let S = rbound ( 0) ( 1) ( 0)
        let T = rbound ( 1) ( 0) ( 0)
        in updater C_pow B N W C E S T
        )

let single_iteration_stencil [Nz][Ny][Nx]
    (temp: [Nz][Ny][Nx]f32) (power: [Nz][Ny][Nx]f32)
    updater
    : [Nz][Ny][Nx]f32 =
  let ixs = [(-1,0,0),(0,-1,0),(0,0,-1),(0,0,0),(0,0,1),(0,1,0),(1,0,0)] in
  let f pow v = updater pow v[0] v[1] v[2] v[3] v[4] v[5] v[6] in
  stencil_3d ixs f power temp

let compute_chip_parameters (Nz: i64) (Ny: i64) (Nx: i64) : (f32,f32,f32,f32,f32) =
  let grid_width  = chip_width  / f32.i64(Nx)
  let grid_height = chip_height / f32.i64(Ny)
  let grid_depth  = t_chip      / f32.i64(Nz)
  let cap = factor_chip * spec_heat_si * t_chip * grid_width * grid_height
  let rx = grid_width / (2 * k_si * t_chip * grid_height)
  let ry = grid_height / (2 * k_si * t_chip * grid_width)
  let rz = grid_depth / (k_si * grid_height * grid_width)
  let max_slope = max_pd / (factor_chip * t_chip * spec_heat_si)
  let step = precision / max_slope
  let stepDivCap = step / cap
  let cew = stepDivCap / rx
  let cns = stepDivCap / ry
  let ctb = stepDivCap / rz
  let cc = 1.0 - (2.0*cew + 2.0*cns + 3.0*ctb)
  in (stepDivCap, cew, cns, ctb, cc)

let compute_tran_temp [Nz][Ny][Nx]
    (num_iterations: i32) f
    (temp: [Nz][Ny][Nx]f32)
    (power: [Nz][Ny][Nx]f32): [Nz][Ny][Nx]f32 =
  let params = compute_chip_parameters Nz Ny Nx
  let update_fun = calculate_update params
  let single_iter arr = f arr power update_fun
  in iterate num_iterations single_iter temp

let num_iterations: i32 = 5
module hotspot3dCommon = {
  let bench_maps    = compute_tran_temp num_iterations single_iteration_maps
  let bench_stencil = compute_tran_temp num_iterations single_iteration_stencil
}
