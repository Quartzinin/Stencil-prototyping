import "./common/hotspot2dCommon"
-- ==
-- entry: main
-- compiled random input { [4096][4096]f32 } auto output
-- compiled random input { [8192][8192]f32 } auto output
entry main arr = hotspot2dCommon.bench_maps arr arr

