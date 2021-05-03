

let boundc (s:i64) (m:i64) =
  let tmp = if s < 0 then 0 else s in if tmp > m then m else tmp
let boundm (s:i64) (m:i64) = i64.min (i64.max 0 s) m

module edgeHandling = {

--  let extendEdge2D arr (maxy:i64) (maxx:i64) (y:i64) (x:i64) =
--    let by = boundc y maxy
--    let bx = boundc x maxx
--    in #[unsafe] arr[by,bx]
--  let extendEdge3D arr (maxz:i64) (maxy:i64) (maxx:i64) (z:i64) (y:i64) (x:i64) =
--    let bz = boundc z maxz
--    let by = boundc y maxy
--    let bx = boundc x maxx
--    in #[unsafe] arr[bz,by,bx]

 let extendEdge2D arr (maxy:i64) (maxx:i64) (y:i64) (x:i64) =
    let by = boundm y maxy
    let bx = boundm x maxx
    in #[unsafe] arr[by,bx]
 let extendEdge3D arr (maxz:i64) (maxy:i64) (maxx:i64) (z:i64) (y:i64) (x:i64) =
    let bz = boundm z maxz
    let by = boundm y maxy
    let bx = boundm x maxx
    in #[unsafe] arr[bz,by,bx]
}
