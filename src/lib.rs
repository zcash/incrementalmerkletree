#![feature(rustc_private)]
extern crate rustc;
extern crate rand;

mod sha256;

trait Hashable: Clone + Copy {
    fn combine(&Self, &Self) -> Self;
    fn blank() -> Self;
}

#[derive(Clone)]
struct IncrementalMerkleTree<T: Hashable> {
    cursor: Leaf<T>,
    depth: usize
}

#[derive(Clone)]
enum Leaf<T: Hashable> {
    Left{parent: Parent<T>, content: T},
    Right{parent: Parent<T>, left: T, content: T} 
}

#[derive(Clone)]
enum Parent<T: Hashable> {
    Empty,
    Left{parent: Box<Parent<T>>},
    Right{left: T, parent: Box<Parent<T>>}
}

impl<T: Hashable> Parent<T> {
    fn ascend<'a, F: FnMut(Option<&'a T>) -> bool>(&'a self, mut cb: F) {
        match *self {
            Parent::Empty => {
                if cb(None) {
                    self.ascend(cb);
                }
            },
            Parent::Left{ref parent} => {
                if cb(None) {
                    parent.ascend(cb);
                }
            },
            Parent::Right{ref left, ref parent} => {
                if cb(Some(left)) {
                    parent.ascend(cb);
                }
            }
        }
    }

    fn advance(self, hash: T) -> Parent<T> {
        match self {
            Parent::Empty => {
                Parent::Right {
                    left: hash,
                    parent: Box::new(Parent::Empty)
                }
            },
            Parent::Left{parent} => {
                Parent::Right {
                    left: hash,
                    parent: parent
                }
            },
            Parent::Right{left, parent} => {
                Parent::Left{
                    parent: Box::new(parent.advance(T::combine(&left, &hash)))
                }
            }
        }
    }
}

impl<T: Hashable> IncrementalMerkleTree<T> {
    fn new(d: usize, initial: T) -> IncrementalMerkleTree<T> {
        assert!(d != 0);

        IncrementalMerkleTree {
            cursor: Leaf::Left{parent: Parent::Empty, content: initial},
            depth: d
        }
    }

    fn completed(&self) -> bool {
        match self.cursor {
            Leaf::Left{..} => false,
            Leaf::Right{ref parent, ..} => {
                let complete = &mut true;
                let depth = &mut (self.depth - 1);

                parent.ascend(|left| {
                    if *depth == 0 {
                        return false;
                    }

                    if left.is_none() {
                        *complete = false;
                        return false;
                    }

                    *depth -= 1;

                    true
                });

                *complete
            }
        }
    }

    fn append(self, obj: T) -> IncrementalMerkleTree<T> {
        match self.cursor {
            Leaf::Left{parent, content} => {
                IncrementalMerkleTree {
                    cursor: Leaf::Right{
                        parent: parent,
                        left: content,
                        content: obj
                    },
                    depth: self.depth
                }
            },
            Leaf::Right{parent, left, content} => {
                IncrementalMerkleTree {
                    cursor: Leaf::Left{
                        parent: parent.advance(T::combine(&left, &content)),
                        content: obj
                    },
                    depth: self.depth
                }
            }
        }
    }

    fn unfilled(&self, mut skip: usize) -> usize {
        let parent = match self.cursor {
            Leaf::Left{ref parent, ..} => {
                if skip == 0 {
                    return 0;
                } else {
                    skip -= 1;

                    parent
                }
            },
            Leaf::Right{ref parent, ..} => {
                parent
            }
        };

        let mut depth = &mut 0;
        parent.ascend(|left| {
            *depth += 1;

            match left {
                Some(_) => {
                    return true;
                },
                None => {
                    if skip == 0 {
                        return false;
                    } else {
                        skip -= 1;
                        return true;
                    }
                }
            }
        });

        return *depth;
    }

    fn root(&self) -> T {
        self.root_advanced(Some(T::blank()).iter().cycle())
    }

    fn root_advanced<'a, I: Iterator<Item=&'a T>>(&self, mut it: I) -> T where T: 'a {
        let (parent, mut child) = match self.cursor {
            Leaf::Left{ref parent, ref content} => {
                (parent, T::combine(content, it.next().unwrap()))
            },
            Leaf::Right{ref parent, ref left, ref content} => {
                (parent, T::combine(left, content))
            }
        };

        let mut depth = self.depth - 1;
        {
            let child = &mut child;

            parent.ascend(move |left| {
                if depth == 0 {
                    return false;
                }

                match left {
                    Some(left) => {
                        *child = T::combine(left, &*child);
                    },
                    None => {
                        *child = T::combine(&*child, it.next().unwrap());
                    }
                }

                depth = depth - 1;

                true
            });
        }

        return child;
    }
}

#[derive(Clone)]
struct IncrementalWitness<T: Hashable> {
    tree: IncrementalMerkleTree<T>,
    delta: IncrementalDelta<T>
}

#[derive(Clone)]
struct IncrementalDelta<T: Hashable> {
    filled: Vec<T>,
    active: Option<IncrementalMerkleTree<T>>
}

impl<T: Hashable> IncrementalWitness<T> {
    fn new(from: &IncrementalMerkleTree<T>) -> IncrementalWitness<T> {
        IncrementalWitness {
            tree: from.clone(),
            delta: IncrementalDelta {
                filled: vec![],
                active: None
            }
        }
    }

    fn append(&mut self, object: T) {
        match self.delta.active.take() {
            Some(active) => {
                let active = active.append(object);

                if active.completed() {
                    self.delta.filled.push(active.root());
                } else {
                    self.delta.active = Some(active);
                }
            },
            None => {
                match self.tree.unfilled(self.delta.filled.len()) {
                    0 => {
                        self.delta.filled.push(object);
                    },
                    i => {
                        self.delta.active = Some(IncrementalMerkleTree::new(i, object));
                    }
                }
            }
        }
    }

    fn root(&self) -> T {
        self.tree.root_advanced(self.delta.filled.iter() // use filled values
                                .chain(self.delta.active.as_ref().map(|x| x.root()).as_ref()) // then use the active root
                                .chain(Some(T::blank()).iter().cycle())) // then fill in with blanks
    }
}


mod test {
    use super::{IncrementalMerkleTree, IncrementalWitness};
    use super::sha256::*;

    #[test]
    fn test_root() {
        let a = Sha256Digest::rand(0);

        let tree = IncrementalMerkleTree::new(3, a);

        assert_eq!(tree.root(), Sha256Digest([94, 162, 216, 229, 230, 128, 153, 35, 89, 40, 180, 159, 125, 27, 48, 80, 181, 73, 7, 195, 182, 223, 83, 165, 59, 200, 234, 181, 106, 3, 243, 228]));

        let b = Sha256Digest::rand(1);

        let tree = tree.append(b);

        assert_eq!(tree.root(), Sha256Digest([222, 23, 196, 222, 130, 80, 115, 139, 134, 72, 108, 150, 235, 75, 216, 5, 63, 101, 2, 237, 51, 47, 165, 216, 40, 15, 209, 176, 10, 192, 224, 26]));
    }

    #[test]
    fn test_unfilled() {
        let a = Sha256Digest::rand(0);
        let mut tree = IncrementalMerkleTree::new(3, a);

        for i in 0..4 {
            let b = Sha256Digest::rand(i+1);
            tree = tree.append(b);
        }

        assert_eq!(tree.unfilled(0), 0);
        assert_eq!(tree.unfilled(1), 1);
        assert_eq!(tree.unfilled(2), 3);
    }

    #[test]
    fn test_complete() {
        let a = Sha256Digest::rand(0);
        let mut tree = IncrementalMerkleTree::new(3, a);

        for i in 0..7 {
            assert_eq!(tree.completed(), false);
            let b = Sha256Digest::rand(i+1);
            tree = tree.append(b);
        }

        assert_eq!(tree.completed(), true);
    }

    #[test]
    fn test_witness() {
        let a = Sha256Digest::rand(0);
        let mut tree = IncrementalMerkleTree::new(3, a);

        let mut witness = IncrementalWitness::new(&tree);

        assert_eq!(tree.root(), witness.root());
        assert_eq!(witness.delta.filled.len(), 0);
        assert!(witness.delta.active.is_none());

        for i in 1..8 {
            let b = Sha256Digest::rand(i);
            witness.append(b);
            tree = tree.append(b);

            match i {
                1 => {
                    assert_eq!(witness.delta.filled.len(), 1);
                    assert!(witness.delta.active.is_none());
                },
                i if i <= 2 => {
                    assert_eq!(witness.delta.filled.len(), 1);
                    assert!(witness.delta.active.is_some());
                },
                i if i == 3 => {
                    assert_eq!(witness.delta.filled.len(), 2);
                    assert!(witness.delta.active.is_none());
                },
                i if i < 7 => {
                    assert_eq!(witness.delta.filled.len(), 2);
                    assert!(witness.delta.active.is_some());
                },
                a @ _ => {
                    assert_eq!(a, 7);
                    assert_eq!(witness.delta.filled.len(), 3);
                    assert!(witness.delta.active.is_none());
                }
            }

            assert_eq!(tree.root(), witness.root());
        }
    }
}