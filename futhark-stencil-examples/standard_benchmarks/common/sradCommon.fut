-- The computes the SRAD-v2 kernel from the Rodinia benchmarks.
import "./edgeHandling"

-- code based on
-- https://github.com/diku-dk/futhark-benchmarks/blob/master/rodinia/srad/srad.fut

let stencil_body_fun1
    (std_dev: f32)
    ((N, W, C, E, S): (f32,f32,f32,f32,f32))
    : f32 =
  let dN_k = N / C
  let dS_k = S / C
  let dW_k = W / C
  let dE_k = E / C
  let g2 = (dN_k*dN_k + dS_k*dS_k +
            dW_k*dW_k + dE_k*dE_k) / (C*C)
  let l = (dN_k + dS_k + dW_k + dE_k) / C
  let num = (0.5*g2) - ((1.0/16.0)*(l*l))
  let den = 1.0 + 0.25*l
  let qsqr = num / (den*den)
  let den = (qsqr-std_dev) / (std_dev * (1.0+std_dev))
  let c_k = 1.0 / (1.0+den)
  let c_k = f32.max 0.0 (f32.min 1.0 c_k)
  in c_k

let stencil_body_fun2
    (lambda: f32)
    (((_, pN), (_, pW), (cC,pC), (cE,pE), (cS,pS))
    :((f32,f32),(f32,f32),(f32,f32),(f32,f32),(f32,f32)))
    : f32 =
  let dN_k = pN / pC
  let dS_k = pS / pC
  let dW_k = pW / pC
  let dE_k = pE / pC
  let cN = cC
  let cW = cC
  let d = cN*dN_k + cS*dS_k + cW*dW_k + cE*dE_k
  in pC + 0.25 * lambda * d

let update_fun_maps [Ny][Nx]
    (std_dev: f32) (lambda: f32) (image: [Ny][Nx]f32)
    : [Ny][Nx]f32 =
  let bound arr = edgeHandling.extendEdge2D arr (Ny-1) (Nx-1)
  let tmp_image = tabulate_2d Ny Nx (\r c ->
        let rbound = bound image r c
        let N = rbound (-1) ( 0)
        let W = rbound ( 0) (-1)
        let C = rbound ( 0) ( 0)
        let E = rbound ( 0) ( 1)
        let S = rbound ( 1) ( 0)

        in stencil_body_fun1 std_dev (N, W, C, E, S)
      )
  let zip_image_tmp = map2 zip tmp_image image
  let image = tabulate_2d Ny Nx (\r c ->
        let rbound = bound zip_image_tmp r c
        let N = rbound (-1) ( 0)
        let W = rbound ( 0) (-1)
        let C = rbound ( 0) ( 0)
        let E = rbound ( 0) ( 1)
        let S = rbound ( 1) ( 0)

        in stencil_body_fun2 lambda (N, W, C, E, S)
      )
  in image

let update_fun_stencil [Ny][Nx]
    (std_dev: f32) (lambda: f32) (image: [Ny][Nx]f32)
    : [Ny][Nx]f32 =
  let ixs = [(-1,0),(0,-1),(0,0),(0,1),(1,0)] in
  let fun1 _ vars = stencil_body_fun1 std_dev (vars[0], vars[1], vars[2], vars[3], vars[4]) in
  let fun2 _ vars = stencil_body_fun2 lambda  (vars[0], vars[1], vars[2], vars[3], vars[4]) in
  let empty = map (map (const ())) image in
  stencil_2d ixs fun1 empty image
    |> flip (map2 zip) image
    |> stencil_2d ixs fun2 empty

let do_srad [Ny][Nx] (niter: i32) (lambda: f32) f (image: [Ny][Nx]u8): [Ny][Nx]f32 =
  let flat_length_f32: f32 = f32.i64 (Ny * Nx)
  let image = map (map (f32.u8 >-> (/ 255.0) >-> f32.exp)) image
  let update_fun image =
    let sum = f32.sum (flatten image)
    let sum_sq = f32.sum (map (\x -> x*x) (flatten image))
    let mean = sum / flat_length_f32
    let variance = (sum_sq / flat_length_f32) - mean*mean
    let std_dev = variance / (mean*mean)
    in f std_dev lambda image
  let image = iterate niter update_fun image

  let image = map (map (f32.log >-> (* 255.0))) image
  in image

let lambda: f32 = 0.5
let niter: i32 = 10
module sradCommon = {
  let bench_maps = do_srad niter lambda update_fun_maps
  let bench_stencil = do_srad niter lambda update_fun_stencil
}

