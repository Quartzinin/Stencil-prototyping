-- code and comments based on
-- https://github.com/diku-dk/futhark-benchmarks/blob/master/rodinia/srad/srad.fut
--
-- ==
-- entry: bench_maps
-- random input { [458][504]u8 }

-- ==
-- entry: bench_stencils
-- random input { [458][504]u8 }

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

let update_fun_maps [len_y][len_x]
    (std_dev: f32) (lambda: f32) (image: [len_y][len_x]f32)
    : [len_y][len_x]f32 =
  let bound arr r c =
    let br = i64.min (i64.max 0 r) (len_y-1)
    let bc = i64.min (i64.max 0 c) (len_x-1)
    in arr[br,bc]
  let tmp_image = tabulate_2d len_y len_x (\r c->
        let N = bound image (r-1) (c  )
        let W = bound image (r  ) (c-1)
        let C = bound image (r  ) (c  )
        let E = bound image (r  ) (c+1)
        let S = bound image (r+1) (c  )

        in stencil_body_fun1 std_dev (N, W, C, E, S)
      )
  let zip_image_tmp = map2 zip tmp_image image
  let image = tabulate_2d len_y len_x (\r c ->
        let N = bound zip_image_tmp (r-1) (c  )
        let W = bound zip_image_tmp (r  ) (c-1)
        let C = bound zip_image_tmp (r  ) (c  )
        let E = bound zip_image_tmp (r  ) (c+1)
        let S = bound zip_image_tmp (r+1) (c  )

        in stencil_body_fun2 lambda (N, W, C, E, S)
      )
  in image

let update_fun_stencil [len_y][len_x]
    (std_dev: f32) (lambda: f32) (image: [len_y][len_x]f32)
    : [len_y][len_x]f32 =
  let ixs = [(-1,0),(0,-1),(0,0),(0,1),(1,0)] in
  let fun1 _ vars = stencil_body_fun1 std_dev (vars[0], vars[1], vars[2], vars[3], vars[4]) in
  let fun2 _ vars = stencil_body_fun2 lambda  (vars[0], vars[1], vars[2], vars[3], vars[4]) in
  let empty = map (map (const ())) image in
  stencil_2d ixs fun1 empty image
    |> flip (map2 zip) image
    |> stencil_2d ixs fun2 empty

let do_srad [len_y][len_x] (niter: i32) (lambda: f32) f (image: [len_y][len_x]u8): [len_y][len_x]f32 =
  let flat_length_f32: f32 = f32.i64 (len_y * len_x)
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
entry bench_maps = do_srad niter lambda update_fun_maps
entry bench_stencils = do_srad niter lambda update_fun_stencil

