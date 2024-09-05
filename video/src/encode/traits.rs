use std::mem::size_of;
use std::slice::{from_raw_parts, from_raw_parts_mut};

pub trait CastSlice<From>: AsRef<[From]> {
    #[inline]
    fn cast<To>(&self) -> &[To] {
        let slice = self.as_ref();
        unsafe {
            from_raw_parts(
                slice as *const _ as *const To,
                slice.len() * size_of::<From>() / size_of::<To>(),
            )
        }
    }

    #[inline]
    fn cast_mut<To>(&mut self) -> &mut [To] {
        let slice = self.as_ref();
        unsafe {
            from_raw_parts_mut(
                slice as *const _ as *mut To,
                slice.len() * size_of::<From>() / size_of::<To>(),
            )
        }
    }
}

impl<From> CastSlice<From> for &[From] {}
impl<From> CastSlice<From> for [From] {}
