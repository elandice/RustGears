use rustorch::prelude::*;
use std::io::ErrorKind;

struct State(Vec<(String, Tensor)>);

impl Module for State {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.clone()
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        self.0.clone()
    }
}

#[test]
fn checkpoint_retries_after_failed_save_and_rejects_nonfinite_scores() {
    let unique = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let directory = std::env::temp_dir().join(format!("rustorch_callback_{}_{}", std::process::id(), unique));
    let path = directory.join("model.rtsc");
    let model = State(vec![("weight".into(), Tensor::ones(&[2]))]);
    let mut checkpoint = ModelCheckpoint::new(&path);
    assert_eq!(checkpoint.step(f32::NAN, &model).unwrap_err().kind(), ErrorKind::InvalidInput);
    assert_eq!(checkpoint.best_score(), None);
    assert!(checkpoint.step(1.0, &model).is_err());
    assert_eq!(checkpoint.best_score(), None);
    std::fs::create_dir(&directory).unwrap();
    assert!(checkpoint.step(1.0, &model).unwrap());
    assert!(path.exists());
    for score in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(checkpoint.step(score, &model).unwrap_err().kind(), ErrorKind::InvalidInput);
        assert_eq!(checkpoint.best_score(), Some(1.0));
    }
    assert!(!checkpoint.step(1.0, &model).unwrap());
    std::fs::remove_file(path).unwrap();
    std::fs::remove_dir(directory).unwrap();
}

#[test]
fn early_stopping_nonfinite_scores_do_not_poison_best_snapshot() {
    let model = BatchNorm1d::new(2);
    let mut stopping = EarlyStopping::new(2);
    assert!(!stopping.step(f32::NAN, &model));
    assert_eq!(stopping.best_score(), None);
    assert_eq!(stopping.counter(), 1);
    assert!(stopping.step(1.0, &model));
    assert_eq!(stopping.counter(), 0);
    model.gamma.copy_from_slice(&[9.0, 9.0]);
    model.running_mean.copy_from_slice(&[7.0, 7.0]);
    assert!(!stopping.step(f32::INFINITY, &model));
    assert!(!stopping.step(f32::NEG_INFINITY, &model));
    assert!(stopping.should_stop());
    assert_eq!(stopping.best_score(), Some(1.0));
    assert_eq!(model.gamma.to_vec(), vec![1.0, 1.0]);
    assert_eq!(model.running_mean.to_vec(), vec![0.0, 0.0]);
}

#[test]
fn early_stopping_restore_validates_every_name_and_shape_before_copying() {
    let original = State(vec![
        ("a".into(), Tensor::from_slice(&[1.0, 2.0], &[2])),
        ("b".into(), Tensor::from_slice(&[3.0, 4.0], &[2, 1])),
    ]);
    let mut stopping = EarlyStopping::new(2);
    assert!(stopping.step(1.0, &original));
    for (name, shape) in [("b", vec![1, 2]), ("missing", vec![2, 1]), ("a", vec![2, 1])] {
        let a = Tensor::from_slice(&[8.0, 9.0], &[2]);
        let b = Tensor::from_slice(&[6.0, 7.0], &shape);
        let incompatible = State(vec![("a".into(), a.clone()), (name.into(), b.clone())]);
        assert!(!stopping.restore(&incompatible));
        assert_eq!(a.to_vec(), vec![8.0, 9.0]);
        assert_eq!(b.to_vec(), vec![6.0, 7.0]);
    }
    assert!(!stopping.restore(&State(vec![])));
    let a = Tensor::zeros(&[2]);
    let b = Tensor::zeros(&[2, 1]);
    let reordered = State(vec![("b".into(), b.clone()), ("a".into(), a.clone())]);
    assert!(stopping.restore(&reordered));
    assert_eq!(a.to_vec(), vec![1.0, 2.0]);
    assert_eq!(b.to_vec(), vec![3.0, 4.0]);
}

#[test]
fn early_stopping_requires_finite_min_delta() {
    for delta in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(std::panic::catch_unwind(|| EarlyStopping::new(2).min_delta(delta)).is_err());
    }
}
