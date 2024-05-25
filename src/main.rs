mod value;

use value::Value;

fn main() {
    // let a = Value { data: 10, _grad: 0.0, };
    let a = Value::new_v(10.);
    println!("{}", a.data);
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;
    use crate::value::{_backward2, special_add, special_mul, special_tanh, Value};

    // #[test]
    // fn test_exp() {
    //     let a = Value::new_v(10.);
    //     let b = Value::new_v(-15.);
    //     let c = Value::new_v(2.);
    //     let d = Value::new_v(3.);
    //
    //     let exp = a * b + c / d;
    //     assert_eq!(exp.data, -149.33333);
    //
    //     let a = Value::new_v(10.);
    //     let b = Value::new_v(-15.);
    //
    //     let exp = a + b;
    //     assert_eq!(exp.data, -5.);
    //     assert_eq!(exp._op, "+".to_string());
    // }

    // #[test]
    // fn test_eq() {
    //     let em_op = "".to_string();
    //
    //     let child1 = Value::new_v(10.);
    //     let child2 = Value::new_v(10.);
    //     let a = Value::new_op(3., vec![child1], em_op.clone());
    //     let b = Value::new_op(3., vec![child2], em_op.clone());
    //
    //     assert_eq!(a.data, b.data);
    //
    //     // let child1 = Value::new_v(7.);
    //     // let child2 = Value::new_v(8.);
    //     // let a = Value { data: 3., _grad: 0., _prev: vec![ child1 ], _op: em_op.clone()};
    //     // let b = Value { data: 3., _grad: 0., _prev: vec![ child2 ], _op: em_op.clone()};
    //     // assert_ne!(a, b);
    //
    //     // let child1 = Value::new_v(8.);
    //     // let child2 = Value::new_v(8.);
    //     // let a = Value { data: 3., _grad: 0., _prev: vec![ child1 ], _op: em_op.clone()};
    //     // let b = Value { data: 3., _grad: 0., _prev: vec![ child2 ], _op: em_op.clone()};
    //     // this won't work for now but let's assume it does.
    //     // assert_ne!(a, b);
    // }

    #[test]
    fn neuron() {
        let x1 = Rc::new(RefCell::new(Value::new_v(3.0)));
        let x2 = Rc::new(RefCell::new(Value::new_v(0.5)));

        let w1 = Rc::new(RefCell::new(Value::new_v(-3.0)));
        let w2 = Rc::new(RefCell::new(Value::new_v(1.0)));

        let b = Rc::new(RefCell::new(Value::new_v(8.0)));

        let l1 = special_mul(x1.clone(), w1.clone());
        let l2 = special_mul(x2.clone(), w2.clone());

        let la = special_add(l1.clone(), l2.clone());

        let n = special_add(la, b.clone());

        let o = special_tanh(n.clone());

        assert_eq!(n.borrow().data, -0.5);
        assert_eq!(o.borrow().data, -0.4620764);
    }

    #[test]
    fn neuron_grad() {
        let x1 = Rc::new(RefCell::new(Value::new_v(2.0)));
        let x2 = Rc::new(RefCell::new(Value::new_v(0.0)));

        let w1 = Rc::new(RefCell::new(Value::new_v(-3.0)));
        let w2 = Rc::new(RefCell::new(Value::new_v(1.0)));

        let b = Rc::new(RefCell::new(Value::new_v(6.8813735870195432)));

        let l1 = special_mul(x1.clone(), w1.clone());
        let l2 = special_mul(x2.clone(), w2.clone());

        let la = special_add(l1.clone(), l2.clone());

        let n = special_add(la.clone(), b.clone());

        let o = special_tanh(n.clone());


        o.borrow_mut()._grad = 1.;

        /**
        o.grad = 1

        n.grad == 0.5 (or rounded)

        x1w1x2w2.gra = 0.5 // la
        b.grad = 0.5

        x1w1 = 0.5 // l1
        x2w2 = 0.5  // l2

        x2.grad = w2.data * x2w2.grad = 0.5
        w2.grad = x2.data * x2w2.grad = 0

        x1.grad = w1.data * x1w1.grad = -1.5
        w1.grad = x1.data * x1w1.grad = 1
        **/

        o.borrow_mut()._backward();
        assert_eq!(n.borrow()._grad, 0.50006473);

        n.borrow_mut()._backward();
        assert_eq!(la.borrow()._grad, 0.50006473);
        assert_eq!(b.borrow()._grad, 0.50006473);

        la.borrow_mut()._backward();
        assert_eq!(l1.borrow()._grad, 0.50006473);
        assert_eq!(l2.borrow()._grad,  0.50006473);

        l1.borrow_mut()._backward();
        assert_eq!(x1.borrow()._grad, -1.5001942);
        assert_eq!(w1.borrow()._grad, 1.0001295);

        l2.borrow_mut()._backward();
        assert_eq!(x2.borrow()._grad, 0.50006473);
        assert_eq!(w2.borrow()._grad, 0.);
    }

    #[test]
    fn neuron_grad2() {
        let x1 = Rc::new(RefCell::new(Value::new_v(2.0)));
        let x2 = Rc::new(RefCell::new(Value::new_v(0.0)));

        let w1 = Rc::new(RefCell::new(Value::new_v(-3.0)));
        let w2 = Rc::new(RefCell::new(Value::new_v(1.0)));

        let b = Rc::new(RefCell::new(Value::new_v(6.8813735870195432)));

        let mut l1 = special_mul(x1.clone(), w1.clone());
        let mut l2 = special_mul(x2.clone(), w2.clone());

        let mut la = special_add(l1.clone(), l2.clone());

        let mut n = special_add(la.clone(), b.clone());

        let mut o = special_tanh(n.clone());


        o.borrow_mut()._grad = 1.;

        /**
        o.grad = 1

        n.grad == 0.5 (or rounded)

        x1w1x2w2.gra = 0.5 // la
        b.grad = 0.5

        x1w1 = 0.5 // l1
        x2w2 = 0.5  // l2

        x2.grad = w2.data * x2w2.grad = 0.5
        w2.grad = x2.data * x2w2.grad = 0

        x1.grad = w1.data * x1w1.grad = -1.5
        w1.grad = x1.data * x1w1.grad = 1
        **/

        _backward2(o.clone());
        assert_eq!(n.clone().borrow()._grad, 0.50006473);

        _backward2(n.clone());
        assert_eq!(la.borrow()._grad, 0.50006473);
        assert_eq!(b.borrow()._grad, 0.50006473);

        _backward2(la.clone());
        assert_eq!(l1.borrow()._grad, 0.50006473);
        assert_eq!(l2.borrow()._grad,  0.50006473);

        _backward2(l1.clone());
        assert_eq!(x1.borrow()._grad, -1.5001942);
        assert_eq!(w1.borrow()._grad, 1.0001295);

        _backward2(l2.clone());
        assert_eq!(x2.borrow()._grad, 0.50006473);
        assert_eq!(w2.borrow()._grad, 0.);
    }
}