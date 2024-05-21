mod value;

use value::Value;

fn main() {
    // let a = Value { data: 10, _grad: 0.0, };
    let a = Value::new_v(10.);
    println!("{}", a.data);
}

#[cfg(test)]
mod tests {
    use crate::value::Value;

    #[test]
    fn test_exp() {
        let a = Value::new_v(10.);
        let b = Value::new_v(-15.);
        let c = Value::new_v(2.);
        let d = Value::new_v(3.);

        let exp = a * b + c / d;
        assert_eq!(exp.data, -149.33333);

        let a = Value::new_v(10.);
        let b = Value::new_v(-15.);

        let exp = a + b;
        assert_eq!(exp.data, -5.);
        assert_eq!(exp._op, "+".to_string());
    }

    #[test]
    fn test_eq() {
        let em_op = "".to_string();

        let child1 = Value::new_v(10.);
        let child2 = Value::new_v(10.);
        let a = Value::new_op(3., vec![child1], em_op.clone());
        let b = Value::new_op(3., vec![child2], em_op.clone());

        assert_eq!(a.data, b.data);

        // let child1 = Value::new_v(7.);
        // let child2 = Value::new_v(8.);
        // let a = Value { data: 3., _grad: 0., _prev: vec![ child1 ], _op: em_op.clone()};
        // let b = Value { data: 3., _grad: 0., _prev: vec![ child2 ], _op: em_op.clone()};
        // assert_ne!(a, b);

        // let child1 = Value::new_v(8.);
        // let child2 = Value::new_v(8.);
        // let a = Value { data: 3., _grad: 0., _prev: vec![ child1 ], _op: em_op.clone()};
        // let b = Value { data: 3., _grad: 0., _prev: vec![ child2 ], _op: em_op.clone()};
        // this won't work for now but let's assume it does.
        // assert_ne!(a, b);
    }

    #[test]
    fn neuron() {
        let x1 = Value::new_v(2.0);
        let x2 = Value::new_v(0.0);

        let w1 = Value::new_v(-3.0);
        let w2 = Value::new_v(1.0);

        let b = Value::new_v(8.0);

        let l1 = x1*w1;
        let l2 = x2*w2;

        let la = l1 + l2;

        let n = la + b;

        let o = n.tanh();

        assert_eq!(o.data, 0.9640129);
    }

    #[test]
    fn nueron_grad() {
        let x1 = Value::new_v(2.0);
        let x2 = Value::new_v(0.0);

        let w1 = Value::new_v(-3.0);
        let w2 = Value::new_v(1.0);

        let b = Value::new_v(6.8813735870195432);

        let l1 = x1*w1;
        let l2 = x2*w2;

        let la = l1 + l2;

        let mut n = la + b;

        // let mut o = n.tanh();

        // assert_eq!(o.data, 0.707061);


        // o._grad = 1.;
        //
        // match o._backward {
        //     f1  => {}
        // }
        // assert_eq!(o.data, 1.)

        n._grad = 1.;
        (n._backward)();
        assert_eq!(n.data, 1.);
    }
}