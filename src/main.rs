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
    use crate::value::{_backward2, backward, special_add, special_mul, special_tanh, Value};

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
    fn neuron_grad2() {
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

        _backward2(o.clone());
        assert_eq!(n.clone().borrow()._grad, 0.50006473);

        _backward2(n.clone());
        assert_eq!(la.clone().borrow()._grad, 0.50006473);
        assert_eq!(b.clone().borrow()._grad, 0.50006473);

        _backward2(la.clone());
        assert_eq!(l1.clone().borrow()._grad, 0.50006473);
        assert_eq!(l2.clone().borrow()._grad,  0.50006473);

        _backward2(l1.clone());
        assert_eq!(x1.borrow()._grad, -1.5001942);
        assert_eq!(w1.borrow()._grad, 1.0001295);

        _backward2(l2.clone());
        assert_eq!(x2.borrow()._grad, 0.50006473);
        assert_eq!(w2.borrow()._grad, 0.);
    }

    #[test]
    fn neuron_grad_topo() {
        let x1 = Rc::new(RefCell::new(Value::new_v(2.0)));
        x1.borrow_mut()._label = "x1".to_string();
        let x2 = Rc::new(RefCell::new(Value::new_v(0.0)));
        x2.borrow_mut()._label = "x2".to_string();

        let w1 = Rc::new(RefCell::new(Value::new_v(-3.0)));
        w1.borrow_mut()._label = "w1".to_string();
        let w2 = Rc::new(RefCell::new(Value::new_v(1.0)));
        w2.borrow_mut()._label = "w2".to_string();

        let b = Rc::new(RefCell::new(Value::new_v(6.8813735870195432)));
        b.borrow_mut()._label = "b".to_string();

        let l1 = special_mul(x1.clone(), w1.clone());
        l1.borrow_mut()._label = "l1".to_string();
        let l2 = special_mul(x2.clone(), w2.clone());
        l2.borrow_mut()._label = "l2".to_string();

        let la = special_add(l1.clone(), l2.clone());
        la.borrow_mut()._label = "la".to_string();

        let n = special_add(la.clone(), b.clone());
        n.borrow_mut()._label = "n".to_string();

        let o = special_tanh(n.clone());
        o.borrow_mut()._label = "o".to_string();

        assert_eq!(o.borrow().data, 0.707061);

        o.borrow_mut()._grad = 1.;

        backward(o.clone());

        assert_eq!(n.clone().borrow()._grad, 0.50006473);

        assert_eq!(la.clone().borrow()._grad, 0.50006473);
        assert_eq!(b.clone().borrow()._grad, 0.50006473);

        assert_eq!(l1.clone().borrow()._grad, 0.50006473);
        assert_eq!(l2.clone().borrow()._grad,  0.50006473);

        assert_eq!(x1.borrow()._grad, -1.5001942);
        assert_eq!(w1.borrow()._grad, 1.0001295);

        assert_eq!(x2.borrow()._grad, 0.50006473);
        assert_eq!(w2.borrow()._grad, 0.);
    }

    #[test]
    fn grad_acc_bug() {
        let a = Rc::new(RefCell::new(Value::new_v(3.0)));
        a.borrow_mut()._label = "a".to_string();

        let b = special_add(a.clone(), a.clone());
        b.borrow_mut()._label = "b".to_string();

        b.borrow_mut()._grad = 1.;
        backward(b.clone());

        assert_eq!(b.borrow().data, 6.);

        assert_eq!(a.borrow().data, 3.);
        assert_eq!(a.borrow()._grad, 2.);
    }

    #[test]
    fn grad_acc_bug2() {
        let a = Rc::new(RefCell::new(Value::new_v(3.0)));
        a.borrow_mut()._label = "a".to_string();

        let b = special_mul(a.clone(), a.clone());
        b.borrow_mut()._label = "b".to_string();

        b.borrow_mut()._grad = 1.;
        backward(b.clone());

        assert_eq!(b.borrow().data, 9.);

        assert_eq!(a.borrow().data, 3.);
        assert_eq!(a.borrow()._grad, 6.);
    }


    #[test]
    fn grad_acc_bug3() {
        let a = Rc::new(RefCell::new(Value::new_v(-2.)));
        a.borrow_mut()._label = "a".to_string();

        let b = Rc::new(RefCell::new(Value::new_v(3.)));
        b.borrow_mut()._label = "b".to_string();

        let d = special_mul(a.clone(), b.clone());
        d.borrow_mut()._label = "d".to_string();

        let e = special_add(a.clone(), b.clone());
        e.borrow_mut()._label = "e".to_string();

        let f = special_mul(d.clone(), e.clone());
        f.borrow_mut()._label = "f".to_string();

        f.borrow_mut()._grad = 1.;
        backward(f.clone());

        assert_eq!(e.borrow().data, 1.);
        assert_eq!(e.borrow()._grad, -6.);

        assert_eq!(d.borrow().data, -6.);
        assert_eq!(d.borrow()._grad, 1.);

        assert_eq!(a.borrow().data, -2.);
        assert_eq!(a.borrow()._grad, -3.);

        assert_eq!(b.borrow().data, 3.);
        assert_eq!(b.borrow()._grad, -8.);
    }

    #[test]
    fn neuron_grad_() {
        let x1 = Rc::new(RefCell::new(Value::new_v(2.0)));
        x1.borrow_mut()._label = "x1".to_string();
        let x2 = Rc::new(RefCell::new(Value::new_v(0.0)));
        x2.borrow_mut()._label = "x2".to_string();

        let w1 = Rc::new(RefCell::new(Value::new_v(-3.0)));
        w1.borrow_mut()._label = "w1".to_string();
        let w2 = Rc::new(RefCell::new(Value::new_v(1.0)));
        w2.borrow_mut()._label = "w2".to_string();

        let b = Rc::new(RefCell::new(Value::new_v(6.8813735870195432)));
        b.borrow_mut()._label = "b".to_string();

        let l1 = special_mul(x1.clone(), w1.clone());
        l1.borrow_mut()._label = "l1".to_string();
        let l2 = special_mul(x2.clone(), w2.clone());
        l2.borrow_mut()._label = "l2".to_string();

        let la = special_add(l1.clone(), l2.clone());
        la.borrow_mut()._label = "la".to_string();

        let n = special_add(la.clone(), b.clone());
        n.borrow_mut()._label = "n".to_string();

        let o = special_tanh(n.clone());
        o.borrow_mut()._label = "o".to_string();

        assert_eq!(o.borrow().data, 0.707061);

        o.borrow_mut()._grad = 1.;

        backward(o.clone());

        assert_eq!(n.clone().borrow()._grad, 0.50006473);

        assert_eq!(la.clone().borrow()._grad, 0.50006473);
        assert_eq!(b.clone().borrow()._grad, 0.50006473);

        assert_eq!(l1.clone().borrow()._grad, 0.50006473);
        assert_eq!(l2.clone().borrow()._grad,  0.50006473);

        assert_eq!(x1.borrow()._grad, -1.5001942);
        assert_eq!(w1.borrow()._grad, 1.0001295);

        assert_eq!(x2.borrow()._grad, 0.50006473);
        assert_eq!(w2.borrow()._grad, 0.);
    }
}