use std::cell::RefCell;
use std::rc::Rc;

#[derive(Clone)]
pub struct Value<T: Copy>
{
    pub _label: String,
    pub data: T,

    pub _grad: T,

    pub _prev: Vec<Rc<RefCell<Value<T>>>>,
    pub _op: String,
}

impl Value<f32> {
    pub fn new() -> Self {
        return Value {
            data: 0., _grad: 0., _prev: vec![], _op: "".to_string(),

            _label: "".to_string(),
        }
    }

    pub fn new_v(val: f32) -> Self {
        let mut v: Value<f32> = Value::new();
        v.data = val;
        v._op = "leaf".to_string();
        v
    }

    pub fn new_op(val: f32, _children: Vec<Rc<RefCell<Value<f32>>>>, _op: String) -> Self {
        let mut v: Value<f32> = Value::new();
        v.data = val;
        v._prev = _children;
        v._op = _op;
        v
    }
}

pub fn backward(slf: Rc<RefCell<Value<f32>>>) {
    let mut topo = Vec::new();
    let mut visited = Vec::new();
    topo_sort(slf, &mut topo, &mut visited);

    for node in topo.iter().rev() {

        println!("backward for {}", node.borrow()._label.to_string());
        _backward2(node.clone());
    }
}

pub fn _backward2(slf: Rc<RefCell<Value<f32>>>) {
    let op = slf.borrow()._op.to_string();
    let node_label = slf.borrow()._label.to_string();

    if slf.borrow()._op == "tanh" {
        tanh_backward2(slf.clone());
    } else if slf.borrow()._op == "**" {
        pow_backward2(slf.clone())
    } else if slf.borrow()._op == "/" {
        div_backward2(slf.clone());
    } else if slf.borrow()._op == "*" {
        mul_backward2(slf.clone());
    } else if slf.borrow()._op == "+" {
        add_backward2(slf.clone());
    } else if slf.borrow()._op == "-" {
        sub_backward2(slf.clone());
    } else if slf.borrow()._op == "tanh" {
        tanh_backward2(slf.clone());
    } else if slf.borrow()._op == "leaf" {
        // leaf node
        return;
    } else {
        panic!("unsupported op node_label: {}, op: {}", node_label, op);
    }
}

pub fn topo_sort(
    slf: Rc<RefCell<Value<f32>>>,
    topo: &mut Vec<Rc<RefCell<Value<f32>>>>,
    visited: &mut Vec<Rc<RefCell<Value<f32>>>>) {

    let slf_label= &slf.borrow()._label;

    for node in visited.iter() {
        if node.borrow()._label.eq(slf_label) {
            return;
        }
    }

    visited.push(slf.clone());

    for child in &slf.clone().borrow()._prev {
        topo_sort(child.clone(), topo, visited);
    }

    topo.push(slf.clone());
}

// Operations
pub fn special_add(a: Rc<RefCell<Value<f32>>>, b: Rc<RefCell<Value<f32>>>)
    -> Rc<RefCell<Value<f32>>> {
    let mut nodes = Vec::new();

    let sum;
    if a.as_ptr() == b.as_ptr() {
        sum = 2. * a.borrow_mut().data;
    } else {
        sum = a.borrow_mut().data + b.borrow_mut().data;
    }

    nodes.push(a.clone());
    nodes.push(b.clone());

    let out = Value::new_op(
        sum,
        nodes,
        "+".to_string(),
    );

    return Rc::new(RefCell::new(out));
}

fn add_backward2(slf: Rc<RefCell<Value<f32>>>) {
    println!("{}", slf.borrow()._grad);
    slf.borrow()._prev[0].borrow_mut()._grad += 1. * slf.borrow()._grad;
    slf.borrow()._prev[1].borrow_mut()._grad += 1. * slf.borrow()._grad;
}

fn sub_backward2(slf: Rc<RefCell<Value<f32>>>) {
    println!("{}", slf.borrow()._grad);
    slf.borrow_mut()._prev[0].borrow_mut()._grad += 1. * slf.borrow()._grad;
    slf.borrow_mut()._prev[1].borrow_mut()._grad += -1. * slf.borrow()._grad;
}

pub fn special_mul(a: Rc<RefCell<Value<f32>>>, b: Rc<RefCell<Value<f32>>>)
    ->  Rc<RefCell<Value<f32>>> {
    let mut nodes = Vec::new();

    let p;
    if a.as_ptr() == b.as_ptr() {
        p = f32::powf(a.borrow_mut().data, 2.);
    } else {
        p = a.borrow_mut().data * b.borrow_mut().data;
    }

    nodes.push(a.clone());
    nodes.push(b.clone());

    let out = Value::new_op(
        p,
        nodes,
        "*".to_string(),
    );

    return Rc::new(RefCell::new(out));
}

fn mul_backward2(slf: Rc<RefCell<Value<f32>>>) {
    let a = &slf.borrow()._prev[0];
    let b = &slf.borrow()._prev[1];

    let grad = slf.borrow()._grad;

    if a.as_ptr() == b.as_ptr() {
        let a_data = a.borrow_mut().data;
        a.borrow_mut()._grad += 2. * a_data * grad;
    } else {
        slf.borrow()._prev[0].borrow_mut()._grad +=
            slf.borrow()._prev[1].borrow_mut().data * grad;

        slf.borrow()._prev[1].borrow_mut()._grad +=
            slf.borrow()._prev[0].borrow_mut().data * grad;
    }
}

fn div_backward2(slf: Rc<RefCell<Value<f32>>>) {
    let a = &slf.borrow()._prev[0];
    let b = &slf.borrow()._prev[1];
    let grad = slf.borrow()._grad;

    if a.as_ptr() == b.as_ptr() {
        a.borrow_mut()._grad = 0.;
    } else {
        slf.borrow_mut()._prev[0].borrow_mut()._grad +=
            slf.borrow_mut()._prev[1].borrow_mut().data * grad;
        slf.borrow_mut()._prev[1].borrow_mut()._grad +=
            slf.borrow_mut()._prev[0].borrow_mut().data * grad;
    }
}

pub fn special_tanh(slf: Rc<RefCell<Value<f32>>>) -> Rc<RefCell<Value<f32>>> {
    let n = slf.borrow_mut().data;
    let t = (f32::powf(2.718, 2. * n) - 1.) / (f32::powf(2.718, 2. * n) + 1.);

    let nodes = vec!(slf.clone());

    let out = Value::new_op(
        t,
        nodes,
        "tanh".to_string(),
    );

    return Rc::new(RefCell::new(out));
}

fn tanh_backward2(slf: Rc<RefCell<Value<f32>>>) {
    let g = (1. - f32::powf(slf.borrow().data, 2.)) * slf.borrow()._grad;

    slf.borrow_mut()._prev[0].borrow_mut()._grad += g;
}

fn pow_backward2(slf: Rc<RefCell<Value<f32>>>) {
    slf.borrow_mut()._prev[0].borrow_mut()._grad +=
        (1. - f32::powf(slf.borrow().data, 2.)) * slf.borrow()._grad
}