# PROJECT_STATE.md (keep under 40 lines)
## Done (task ids + one line each)
P01 repo skeleton (10 packages, pyproject, .gitignore, base.yaml), logo.utils.io + tests
(pre-P01) GT conversion, frame extraction, splits (logo.data.*, scripts/*) + tests
P01b rewrote lost sources utils/{paths,geometry}.py, data/{schema,convert_gt,frames,splits}.py from .pyc;
  re-conversion reproduces all 88 processed GT jsons + 9 split files byte-identically; 25 tests pass
## Decisions / deviations from the paper
- (example) detector: Plan A PP-YOLOE-R small, input 1024x576
- see docs/decisions.md
## Interfaces added beyond the CONTRACT (name, signature, file)
- logo.utils.paths: project_root(), check_dataset(ds), check_split(split,final=False), raw_dir(ds),
  frames_dir(ds,video), frame_path(ds,video,frame_id), gt_path(ds,video), split_file(ds,split),
  stage_dir(exp_id,stage,ds)  [src/logo/utils/paths.py]
- logo.utils.geometry: signed_area, is_self_intersecting, order_quad, quad_iou(q1,q2)  [geometry.py]
- logo.data.schema: validate_gt(gt), read_split(path), write_split(path,videos)
- logo.data.convert_gt.*, logo.data.frames.extract_frames(video_path,out_dir,quality=95,overwrite=False),
  logo.data.splits.train_val_split(videos,val_frac,seed)
## Open problems
- detector{} config keys not yet defined (filled in P02)
## Next task
P02
