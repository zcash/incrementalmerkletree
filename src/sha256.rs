use rustc::util::sha2::{Digest,Sha256};
use std::u8;

use super::Hashable;

impl Hashable for Sha256Digest {
    fn combine(left: &Self, right: &Self) -> Sha256Digest {
        sha256_compression_function(Sha256Block::new_from_digests(left, right))
    }

    fn blank() -> Sha256Digest {
        Sha256Digest([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    }
}

struct Sha256Block(pub [u8; 64]);
impl Sha256Block {
    fn new_from_digests(left: &Sha256Digest, right: &Sha256Digest) -> Sha256Block {
        use std::mem;

        struct CompoundDigest {
            left: Sha256Digest,
            right: Sha256Digest
        }

        let compound = CompoundDigest { left: *left, right: *right };

        unsafe { mem::transmute(compound) }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct Sha256Digest(pub [u8; 32]);

impl Sha256Digest {
    pub fn rand(seed: usize) -> Sha256Digest {
        use rand::{self,Rng,SeedableRng,StdRng};

        let seed: [usize; 1] = [seed];

        let mut rng = StdRng::from_seed(&seed);

        Sha256Digest([rng.gen(), rng.gen(), rng.gen(), rng.gen(), rng.gen(), rng.gen(), rng.gen(), rng.gen(),
                      rng.gen(), rng.gen(), rng.gen(), rng.gen(), rng.gen(), rng.gen(), rng.gen(), rng.gen(),
                      rng.gen(), rng.gen(), rng.gen(), rng.gen(), rng.gen(), rng.gen(), rng.gen(), rng.gen(),
                      rng.gen(), rng.gen(), rng.gen(), rng.gen(), rng.gen(), rng.gen(), rng.gen(), rng.gen()
                     ])
    }
}

// todo: this is not a compression function
fn sha256_compression_function(block: Sha256Block) -> Sha256Digest {
    let mut hash = Sha256::new();

    hash.input(&block.0);

    let res = hash.result_bytes();

    let mut s = Sha256Digest([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

    unsafe {
        use std::ptr;

        ptr::copy_nonoverlapping::<u8>(&res[0], &mut (s.0)[0], 32);
    }

    s
}