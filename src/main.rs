mod value;

use value::Value;

fn main() {
    // let a = Value { data: 10, _grad: 0.0, };
    let a = Value::new_v(10.);
    println!("{:?}", a);
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
    }

    #[test]
    fn test_eq() {
        let child = Value::new_v(10.);
        let a = Value { data: 3., _grad: 0., _prev: vec![ child.clone() ]};
        let b = Value { data: 3., _grad: 0., _prev: vec![ child.clone() ]};

        assert_eq!(a, b);

        let child1 = Value::new_v(7.);
        let child2 = Value::new_v(8.);
        let a = Value { data: 3., _grad: 0., _prev: vec![ child1 ]};
        let b = Value { data: 3., _grad: 0., _prev: vec![ child2 ]};

        assert_ne!(a, b);
    }
}