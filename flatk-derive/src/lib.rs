extern crate proc_macro;

mod entity;

use syn::DeriveInput;

#[proc_macro_derive(Entity, attributes(entity))]
pub fn entity(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input: DeriveInput = syn::parse(input).unwrap();

    let gen = entity::impl_entity(&input);

    gen.into()
}
