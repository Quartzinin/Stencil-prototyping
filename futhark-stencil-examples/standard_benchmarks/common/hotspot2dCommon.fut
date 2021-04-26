-- code and comments based on
-- https://github.com/diku-dk/futhark-benchmarks/blob/master/rodinia/hotspot/hotspot.fut

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

-- modified version of rodinia-hotspot update function
-- which assumes explicit edge extension instead of
-- implicit edge extension (aka some E-C, W-C, and such could be filtered out).
let calculate_update (step, cap, rx, ry, rz) C_pow N W C E S : f32 =
  let delta =
        ( (step / cap)
           * (C_pow
              + (((E - C) + (W - C)) / rx)
              + (((S - C) + (N - C)) / ry)
              + (amb_temp - C))
        ) / rz
  in C + delta

let single_iteration_maps [len_y][len_x]
    (temp: [len_y][len_x]f32) (power: [len_y][len_x]f32)
    updater
    : [len_y][len_x]f32 =
  let bound r c =
    let br = i64.min (i64.max 0 r) (len_y-1)
    let bc = i64.min (i64.max 0 c) (len_x-1)
    in #[unsafe] temp[br,bc]
  in tabulate_2d len_y len_x (\r c ->
        let C_pow = power[r,c]
        let N = bound (r-1) (c  )
        let W = bound (r  ) (c-1)
        let C = bound (r  ) (c  )
        let E = bound (r  ) (c+1)
        let S = bound (r+1) (c  )
        in updater C_pow N W C E S
        )

let single_iteration_stencil [len_y][len_x]
    (temp: [len_y][len_x]f32) (power: [len_y][len_x]f32)
    updater
    : [len_y][len_x]f32 =
  let ixs = [(-1,0),(0,-1),(0,0),(0,1),(1,0)] in
  let f pow v = updater pow v[0] v[1] v[2] v[3] v[4] in
  stencil_2d ixs f power temp

let compute_chip_parameters (len_y : i64) (len_x: i64) : (f32,f32,f32,f32,f32) =
  let grid_height = chip_height / f32.i64(len_y)
  let grid_width = chip_width / f32.i64(len_x)
  let cap = factor_chip * spec_heat_si * t_chip * grid_width * grid_height
  let rx = grid_width / (2 * k_si * t_chip * grid_height)
  let ry = grid_height / (2 * k_si * t_chip * grid_width)
  let rz = t_chip / (k_si * grid_height * grid_width)
  let max_slope = max_pd / (factor_chip * t_chip * spec_heat_si)
  let step = precision / max_slope
  in (cap, rx, ry, rz, step)

let compute_tran_temp [len_y][len_x]
    (num_iterations: i32) f (temp: [len_y][len_x]f32) (power: [len_y][len_x]f32): [len_y][len_x]f32 =
  let params = compute_chip_parameters len_y len_x
  let update_fun = calculate_update params
  let single_iter arr = f arr power update_fun
  in iterate num_iterations single_iter temp

let num_iterations: i32 = 5
module hotspot2dCommon = {
  let bench_maps    = compute_tran_temp num_iterations single_iteration_maps
  let bench_stencil = compute_tran_temp num_iterations single_iteration_stencil
}
