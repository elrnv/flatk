use std::collections::{BTreeSet, HashSet};

use lazy_static::lazy_static;
use proc_macro2::{Span, TokenStream};
use proc_macro_crate::crate_name;
use quote::quote;
use syn::*;

//TODO: Add another impl trait for CloneWithStorage

lazy_static! {
    static ref CRATE_NAME: String= {
        // Try to find the crate name in Cargo.toml
        if let Ok(crate_name_str) = crate_name("flatk") {
            crate_name_str
        } else {
            // If we couldn't find it, it could mean either that that flatk-derive was imported without
            // flatk, or that the code is a flatk example. Assume it's an example.
            String::from("flatk")
        }
    };
}

fn crate_name_ident() -> Ident {
    Ident::new(&CRATE_NAME, Span::call_site())
}

/// A helper function to build an set of associated type parameters, from the given Generics.
fn associated_type_params(
    associated_type: Ident,
    generics: &Generics,
    entity_type: &BTreeSet<Ident>,
) -> Vec<Type> {
    // Populate parameters for the associated type
    generics
        .type_params()
        .map(|TypeParam { ident, .. }| {
            if entity_type.contains(ident) {
                parse_quote! { #ident::#associated_type }
            } else {
                parse_quote! { #ident }
            }
        })
        .collect()
}

fn impl_split(ast: &DeriveInput) -> TokenStream {
    let name = &ast.ident;

    let crate_name = crate_name_ident();

    let split_at_impl = impl_simple_trait(
        ast,
        |ty_ident| quote! { #ty_ident: #crate_name::SplitAt },
        |generics,
         ImplInfo {
             entity_field,
             other_field,
             ..
         }| {
            let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
            quote! {
                impl #impl_generics #crate_name::SplitAt for #name #ty_generics #where_clause {
                    fn split_at(self, mid: usize) -> (Self, Self) {
                        let #name {
                            #(
                                #entity_field,
                            )*
                            #(
                                #other_field,
                            )*
                        } = self;

                        #(
                            let #entity_field = #entity_field.split_at(mid);
                        )*
                        (
                            #name {
                                #(
                                    #entity_field: #entity_field.0,
                                )*
                                #(
                                    #other_field: #other_field.clone(),
                                )*
                            },
                            #name {
                                #(
                                    #entity_field: #entity_field.1,
                                )*
                                #(
                                    #other_field,
                                )*
                            },
                        )
                    }
                }
            }
        },
    );

    let split_off_impl = impl_simple_trait(
        ast,
        |ty_ident| quote! { #ty_ident: #crate_name::SplitOff },
        |generics,
         ImplInfo {
             entity_field,
             other_field,
             ..
         }| {
            let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
            quote! {
                impl #impl_generics #crate_name::SplitOff for #name #ty_generics #where_clause {
                    fn split_off(&mut self, mid: usize) -> Self {
                        let #name {
                            #(
                                ref mut #entity_field,
                            )*
                            #(
                                #other_field,
                            )*
                        } = *self;

                        #(
                            let #entity_field = #entity_field.split_off(mid);
                        )*
                        #name {
                            #(
                                #entity_field,
                            )*
                            #(
                                #other_field,
                            )*
                        }
                    }
                }
            }
        },
    );

    let split_prefix_impl = impl_trait(
        ast,
        |_| None,
        |ty_ident, _| quote! { #ty_ident: #crate_name::SplitPrefix<_FlatkN> },
        |_,
         generics,
         ImplInfo {
             entity_field,
             other_field,
             entity_type,
             ..
         },
         _| {
            let sub_params =
                associated_type_params(parse_quote! { Prefix }, &generics, &entity_type);
            let mut extended_generics = generics.clone();
            extended_generics.params.push(parse_quote! { _FlatkN });
            let (impl_generics, _, _) = extended_generics.split_for_impl();
            let (_, ty_generics, where_clause) = generics.split_for_impl();

            quote! {
                impl #impl_generics #crate_name::SplitPrefix<_FlatkN> for #name #ty_generics #where_clause {
                    type Prefix = #name<#(#sub_params,)*>;
                    fn split_prefix(self) -> Option<(Self::Prefix, Self)> {
                        let #name {
                            #(
                                #entity_field,
                            )*
                            #(
                                #other_field,
                            )*
                        } = self;

                        #(
                            let #entity_field = #entity_field.split_prefix()?;
                        )*

                        Some((
                            #name {
                                #(
                                    #entity_field: #entity_field.0,
                                )*
                                #(
                                    #other_field: #other_field.clone(),
                                )*
                            },
                            #name {
                                #(
                                    #entity_field: #entity_field.1,
                                )*
                                #(
                                    #other_field,
                                )*
                            },
                        ))
                    }
                }
            }
        },
    );

    let split_first_impl = impl_simple_trait(
        ast,
        |ty_ident| quote! { #ty_ident: #crate_name::SplitFirst },
        |generics,
         ImplInfo {
             entity_field,
             other_field,
             entity_type,
             ..
         }| {
            let sub_params =
                associated_type_params(parse_quote! { First }, &generics, &entity_type);
            let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
            quote! {
                impl #impl_generics #crate_name::SplitFirst for #name #ty_generics #where_clause {
                    type First = #name<#(#sub_params,)*>;
                    fn split_first(self) -> Option<(Self::First, Self)> {
                        let #name {
                            #(
                                #entity_field,
                            )*
                            #(
                                #other_field,
                            )*
                        } = self;

                        #(
                            let #entity_field = #entity_field.split_first()?;
                        )*

                        Some((
                            #name {
                                #(
                                    #entity_field: #entity_field.0,
                                )*
                                #(
                                    #other_field: #other_field.clone(),
                                )*
                            },
                            #name {
                                #(
                                    #entity_field: #entity_field.1,
                                )*
                                #(
                                    #other_field,
                                )*
                            },
                        ))
                    }
                }
            }
        },
    );

    quote! {
        #split_at_impl
        #split_off_impl
        #split_prefix_impl
        #split_first_impl
    }
}

fn impl_storage(ast: &DeriveInput) -> TokenStream {
    let name = &ast.ident;

    let crate_name = crate_name_ident();

    let into_storage_impl = impl_simple_trait(
        ast,
        |ty_ident| quote! { #ty_ident: #crate_name::IntoStorage },
        |generics,
         ImplInfo {
             entity_field,
             other_field,
             entity_type,
             ..
         }| {
            let sub_params =
                associated_type_params(parse_quote! { StorageType }, &generics, &entity_type);
            let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
            quote! {
                impl #impl_generics #crate_name::IntoStorage for #name #ty_generics #where_clause {
                    type StorageType = #name<#(#sub_params,)*>;
                    fn into_storage(self) -> Self::StorageType {
                        #name {
                            #(
                                #entity_field: self.#entity_field.into_storage(),
                            )*
                            #(
                                #other_field: self.#other_field,
                            )*
                        }
                    }
                }
            }
        },
    );

    let storage_into_impl = impl_trait(
        ast,
        |ty_ident| {
            let alt_type = Ident::new(&format!("{}FlatkAlt", ty_ident), ty_ident.span());
            Some(parse_quote! { #alt_type })
        },
        |ty_ident, alt_ident| quote! { #ty_ident: #crate_name::StorageInto<#alt_ident> },
        |generics,
         ext_generics,
         ImplInfo {
             entity_field,
             other_field,
             entity_type,
             ..
         },
         alt_type_params| {
            let sub_params =
                associated_type_params(parse_quote! { Output }, &generics, &entity_type);
            let (_, ty_generics, _) = generics.split_for_impl();
            let (impl_generics, _, where_clause) = ext_generics.split_for_impl();

            quote! {
                impl #impl_generics #crate_name::StorageInto<#name<#(#alt_type_params,)*>> for #name #ty_generics #where_clause {
                    type Output = #name<#(#sub_params,)*>;
                    #[inline]
                    fn storage_into(self) -> Self::Output {
                        #name {
                            #(
                                #entity_field: self.#entity_field.storage_into(),
                            )*
                            #(
                                #other_field: self.#other_field,
                            )*
                        }
                    }
                }
            }
        },
    );

    let map_storage_impl = impl_simple_trait(
        ast,
        |_| quote! {},
        |generics, _| {
            let mut extended_generics = generics.clone();
            extended_generics.params.push(parse_quote! { _FlatkOut });
            let (impl_generics, _, _) = extended_generics.split_for_impl();
            let (_, ty_generics, where_clause) = generics.split_for_impl();

            quote! {
                impl #impl_generics #crate_name::MapStorage<_FlatkOut> for #name #ty_generics #where_clause {
                    type Input = Self;
                    type Output = _FlatkOut;
                    #[inline]
                    fn map_storage<F: FnOnce(Self) -> _FlatkOut>(self, f: F) -> Self::Output {
                        f(self)
                    }
                }
            }
        },
    );

    let storage_impl = impl_simple_trait(
        ast,
        |ty_ident| quote! { #ty_ident: #crate_name::Storage<Storage = #ty_ident> },
        |generics, _| {
            let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
            quote! {
                impl #impl_generics #crate_name::Storage for #name #ty_generics #where_clause {
                    type Storage = #name #ty_generics;
                    #[inline]
                    fn storage(&self) -> &Self::Storage {
                        self
                    }
                }
            }
        },
    );

    let storage_mut_impl = impl_simple_trait(
        ast,
        |ty_ident| quote! { #ty_ident: #crate_name::StorageMut<Storage = #ty_ident> },
        |generics, _| {
            let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
            quote! {
                impl #impl_generics #crate_name::StorageMut for #name #ty_generics #where_clause {
                    #[inline]
                    fn storage_mut(&mut self) -> &mut Self::Storage {
                        self
                    }
                }
            }
        },
    );

    quote! {
        #into_storage_impl
        #storage_into_impl
        #storage_impl
        #storage_mut_impl
        #map_storage_impl
    }
}

fn impl_get(ast: &DeriveInput) -> TokenStream {
    let name = &ast.ident;

    let crate_name = crate_name_ident();

    let get_index_impl = impl_simple_trait(
        ast,
        |ty_ident| quote! { #ty_ident: #crate_name::Get<'flatk_get, usize> },
        |generics,
         ImplInfo {
             entity_field,
             other_field,
             entity_type,
             ..
         }| {
            let sub_params =
                associated_type_params(parse_quote! { Output }, &generics, &entity_type);
            let mut extended_generics = generics.clone();
            extended_generics.params.push(parse_quote! { 'flatk_get});
            let (impl_generics, _, _) = extended_generics.split_for_impl();
            let (_, ty_generics, where_clause) = generics.split_for_impl();
            quote! {
                impl #impl_generics #crate_name::GetIndex<'flatk_get, #name #ty_generics> for usize  #where_clause {
                    type Output = #name<#(#sub_params,)*>;
                    fn get(self, this: &#name #ty_generics) -> Option<Self::Output> {

                        Some(#name {
                            #(
                                #entity_field: this.#entity_field.get(self)?,
                            )*
                            #(
                                #other_field: this.#other_field,
                            )*
                        })
                    }
                }
            }
        },
    );

    let get_index_for_range_impl = impl_simple_trait(
        ast,
        |ty_ident| quote! { #ty_ident: #crate_name::Get<'flatk_get, std::ops::Range<usize>> },
        |generics,
         ImplInfo {
             entity_field,
             other_field,
             entity_type,
             ..
         }| {
            let sub_params =
                associated_type_params(parse_quote! { Output }, &generics, &entity_type);
            let mut extended_generics = generics.clone();
            extended_generics.params.push(parse_quote! { 'flatk_get});
            let (impl_generics, _, _) = extended_generics.split_for_impl();
            let (_, ty_generics, where_clause) = generics.split_for_impl();
            quote! {
                impl #impl_generics #crate_name::GetIndex<'flatk_get, #name #ty_generics> for ::std::ops::Range<usize>  #where_clause {
                    type Output = #name<#(#sub_params,)*>;
                    fn get(self, this: &#name #ty_generics) -> Option<Self::Output> {

                        Some(#name {
                            #(
                                #entity_field: this.#entity_field.get(self.clone())?,
                            )*
                            #(
                                #other_field: this.#other_field,
                            )*
                        })
                    }
                }
            }
        },
    );

    let get_index_for_static_range_impl = impl_simple_trait(
        ast,
        |ty_ident| quote! { #ty_ident: #crate_name::Get<'flatk_get, #crate_name::StaticRange<N>> },
        |mut generics,
         ImplInfo {
             entity_field,
             other_field,
             entity_type,
             ..
         }| {
            generics
                .make_where_clause()
                .predicates
                .push(parse_quote! { N: #crate_name::Unsigned + Copy });
            let sub_params =
                associated_type_params(parse_quote! { Output }, &generics, &entity_type);
            let mut extended_generics = generics.clone();
            extended_generics.params.push(parse_quote! { N });
            extended_generics.params.push(parse_quote! { 'flatk_get });
            let (impl_generics, _, _) = extended_generics.split_for_impl();
            let (_, ty_generics, where_clause) = generics.split_for_impl();
            quote! {
                impl #impl_generics #crate_name::GetIndex<'flatk_get, #name #ty_generics> for #crate_name::StaticRange<N>  #where_clause {
                    type Output = #name<#(#sub_params,)*>;
                    fn get(self, this: &#name #ty_generics) -> Option<Self::Output> {

                        Some(#name {
                            #(
                                #entity_field: this.#entity_field.get(self)?,
                            )*
                            #(
                                #other_field: this.#other_field,
                            )*
                        })
                    }
                }
            }
        },
    );

    quote! {
        #get_index_impl
        #get_index_for_range_impl
        #get_index_for_static_range_impl
    }
}

fn impl_isolate(ast: &DeriveInput) -> TokenStream {
    let name = &ast.ident;

    let crate_name = crate_name_ident();

    let isolate_index_impl = impl_simple_trait(
        ast,
        |ty_ident| quote! { #ty_ident: #crate_name::Isolate<usize> },
        |generics,
         ImplInfo {
             entity_field,
             other_field,
             entity_type,
             ..
         }| {
            let sub_params =
                associated_type_params(parse_quote! { Output }, &generics, &entity_type);
            let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
            quote! {
                impl #impl_generics #crate_name::IsolateIndex<#name #ty_generics> for usize  #where_clause {
                    type Output = #name<#(#sub_params,)*>;
                    fn try_isolate(self, this: #name #ty_generics) -> Option<Self::Output> {
                        Some(#name {
                            #(
                                #entity_field: #crate_name::Isolate::try_isolate(this.#entity_field, self)?,
                            )*
                            #(
                                #other_field: this.#other_field,
                            )*
                        })
                    }
                }
            }
        },
    );

    let isolate_index_for_range_impl = impl_simple_trait(
        ast,
        |ty_ident| quote! { #ty_ident: #crate_name::Isolate<std::ops::Range<usize>> },
        |generics,
         ImplInfo {
             entity_field,
             other_field,
             entity_type,
             ..
         }| {
            let sub_params =
                associated_type_params(parse_quote! { Output }, &generics, &entity_type);
            let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
            quote! {
                impl #impl_generics #crate_name::IsolateIndex<#name #ty_generics> for ::std::ops::Range<usize>  #where_clause {
                    type Output = #name<#(#sub_params,)*>;
                    fn try_isolate(self, this: #name #ty_generics) -> Option<Self::Output> {

                        Some(#name {
                            #(
                                #entity_field: #crate_name::Isolate::try_isolate(this.#entity_field, self.clone())?,
                            )*
                            #(
                                #other_field: this.#other_field,
                            )*
                        })
                    }
                }
            }
        },
    );

    quote! {
        #isolate_index_impl
        #isolate_index_for_range_impl
    }
}

fn impl_view(ast: &DeriveInput) -> TokenStream {
    let name = &ast.ident;

    let crate_name = crate_name_ident();

    let view_impl = impl_simple_trait(
        ast,
        |ty_ident| quote! { #ty_ident: #crate_name::View<'flatk_view> },
        |generics,
         ImplInfo {
             entity_field,
             other_field,
             entity_type,
             ..
         }| {
            let sub_params = associated_type_params(parse_quote! { Type }, &generics, &entity_type);
            let mut extended_generics = generics.clone();
            extended_generics.params.push(parse_quote! { 'flatk_view });
            let (impl_generics, _, _) = extended_generics.split_for_impl();
            let (_, ty_generics, where_clause) = generics.split_for_impl();
            quote! {
                impl #impl_generics #crate_name::View<'flatk_view> for #name #ty_generics #where_clause {
                    type Type = #name<#(#sub_params,)*>;
                    fn view(&'flatk_view self) -> Self::Type {

                        #name {
                            #(
                                #entity_field: self.#entity_field.view(),
                            )*
                            #(
                                #other_field: self.#other_field,
                            )*
                        }
                    }
                }
            }
        },
    );

    let view_mut_impl = impl_simple_trait(
        ast,
        |ty_ident| quote! { #ty_ident: #crate_name::ViewMut<'flatk_view> },
        |generics,
         ImplInfo {
             entity_field,
             other_field,
             entity_type,
             ..
         }| {
            let sub_params = associated_type_params(parse_quote! { Type }, &generics, &entity_type);

            let mut extended_generics = generics.clone();
            extended_generics.params.push(parse_quote! { 'flatk_view });
            let (impl_generics, _, _) = extended_generics.split_for_impl();
            let (_, ty_generics, where_clause) = generics.split_for_impl();
            quote! {
                impl #impl_generics #crate_name::ViewMut<'flatk_view> for #name #ty_generics #where_clause {
                    type Type = #name<#(#sub_params,)*>;
                    fn view_mut(&'flatk_view mut self) -> Self::Type {

                        #name {
                            #(
                                #entity_field: self.#entity_field.view_mut(),
                            )*
                            #(
                                #other_field: self.#other_field,
                            )*
                        }
                    }
                }
            }
        },
    );

    quote! {
        #view_impl
        #view_mut_impl
    }
}

/// Implement a trait with addition type parameters.
fn impl_trait(
    ast: &DeriveInput,
    mut generate_alt_type_param: impl FnMut(&Ident) -> Option<GenericParam>,
    mut generate_bound: impl FnMut(&Ident, Option<&GenericParam>) -> TokenStream,
    generate_impl: impl FnOnce(&Generics, Generics, ImplInfo, Vec<GenericParam>) -> TokenStream,
) -> TokenStream {
    let impl_info = build_impl_info(ast);

    let mut extended_generics = ast.generics.clone();

    let mut alt_type_params = Vec::new();
    // Here we care about order since alt_type_params will need to be passed in the correct order,
    // so we iterate over the parameters directly.
    for TypeParam {
        ident: ty_ident, ..
    } in ast.generics.type_params()
    {
        if impl_info.entity_type.contains(ty_ident) {
            let alt_type = generate_alt_type_param(ty_ident);
            let bound_tokens = generate_bound(ty_ident, alt_type.as_ref());
            if !bound_tokens.is_empty() {
                extended_generics
                    .make_where_clause()
                    .predicates
                    .push(syn::parse2(bound_tokens).unwrap());
            }
            if let Some(alt_type_param) = alt_type {
                alt_type_params.push(alt_type_param.clone());
                extended_generics.params.push(alt_type_param);
            }
        }
    }

    generate_impl(&ast.generics, extended_generics, impl_info, alt_type_params)
}

/// A select set of information useful for implementing entity traits.
#[derive(Clone, Debug, Default)]
struct ImplInfo {
    pub field_type_attr: Vec<(Ident, Type, bool)>,
    pub entity_field: BTreeSet<Ident>,
    pub other_field: BTreeSet<Ident>,
    pub entity_type: BTreeSet<Ident>,
}

fn build_impl_info(ast: &DeriveInput) -> ImplInfo {
    let type_params: HashSet<_> = ast
        .generics
        .type_params()
        .map(|t| t.ident.clone())
        .collect();

    let mut field_type_attr = Vec::new();
    let mut entity_field = BTreeSet::new();
    let mut other_field = BTreeSet::new();
    let mut entity_type = BTreeSet::new();

    if let Data::Struct(DataStruct {
        fields: Fields::Named(ref fields),
        ..
    }) = ast.data
    {
        for Field {
            ident, ty, attrs, ..
        } in fields.named.iter()
        {
            if let Type::Path(TypePath { qself: None, path }) = ty {
                // Looking for generic parameter that matches one in `type_params`.
                if let Some(ty_ident) = path.get_ident() {
                    if type_params.contains(ty_ident) {
                        entity_field.insert(ident.clone().unwrap());
                        entity_type.insert(ty_ident.clone());
                        field_type_attr.push((ident.clone().unwrap(), ty.clone(), true));
                        continue;
                    }
                }
                // Or we look for an entity attribute for passthrough entities.
                if attrs.contains(&parse_quote! { #[entity] }) {
                    let last = path
                        .segments
                        .last()
                        .expect("Entity field must be a named type");
                    if let PathArguments::AngleBracketed(all_args) = &last.arguments {
                        // All type parameters are assumed to be entity parameters,
                        // Since it's impossible to tell which papameters are significant, without
                        // also knowing the Entity impls of this type.

                        assert!(
                            !all_args.args.is_empty(),
                            "Entity fields must have at least one type parameter"
                        );
                        for arg in all_args.args.iter() {
                            if let GenericArgument::Type(Type::Path(TypePath {
                                qself: None,
                                path,
                            })) = arg
                            {
                                if let Some(ty_ident) = path.get_ident() {
                                    entity_type.insert(ty_ident.clone());
                                }
                            }
                        }
                        entity_field.insert(ident.clone().unwrap());
                        field_type_attr.push((ident.clone().unwrap(), ty.clone(), true));
                        continue;
                    } else {
                        panic!("Entity fields must have at least one type parameter");
                    }
                }
            }
            field_type_attr.push((ident.clone().unwrap(), ty.clone(), false));
            other_field.insert(ident.clone().unwrap());
        }
    }

    ImplInfo {
        field_type_attr,
        entity_field,
        other_field,
        entity_type,
    }
}

/// Implement a trait without addition type parameters.
fn impl_simple_trait(
    ast: &DeriveInput,
    mut generate_bound: impl FnMut(&Ident) -> TokenStream,
    generate_impl: impl FnOnce(Generics, ImplInfo) -> TokenStream,
) -> TokenStream {
    let impl_info = build_impl_info(&ast);

    let mut generics = ast.generics.clone();

    let where_clause = generics.make_where_clause();
    // Note: where clauses don't care about order so we can iterate over the entity_type set
    // directly.
    for ty_ident in &impl_info.entity_type {
        let bound_tokens = generate_bound(ty_ident);
        if !bound_tokens.is_empty() {
            where_clause
                .predicates
                .push(syn::parse2(bound_tokens).unwrap());
        }
    }

    generate_impl(generics, impl_info)
}

pub(crate) fn impl_entity(ast: &DeriveInput) -> TokenStream {
    let name = &ast.ident;

    let crate_name = crate_name_ident();

    let value_type_impl = impl_simple_trait(
        ast,
        |_| TokenStream::new(),
        |generics, _| {
            let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
            quote! {
                impl #impl_generics #crate_name::ValueType for #name #ty_generics #where_clause { }
            }
        },
    );

    let viewed_impl = impl_simple_trait(
        ast,
        |ty_ident| quote! { #ty_ident: #crate_name::Viewed },
        |generics, _| {
            let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
            quote! {
                impl #impl_generics #crate_name::Viewed for #name #ty_generics #where_clause {}
            }
        },
    );

    let set_impl = impl_simple_trait(
        ast,
        |ty_ident| quote! { #ty_ident: #crate_name::Set },
        |generics,
         ImplInfo {
             entity_field,
             entity_type,
             ..
         }| {
            let first_entity_field = entity_field
                .iter()
                .next()
                .expect("Entity types require at least one component");
            let elem_sub_params =
                associated_type_params(parse_quote! { Elem }, &generics, &entity_type);
            let atom_sub_params =
                associated_type_params(parse_quote! { Atom }, &generics, &entity_type);

            let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
            quote! {
                impl #impl_generics #crate_name::Set for #name #ty_generics #where_clause {
                    type Elem = #name<#(#elem_sub_params,)*>;
                    type Atom = #name<#(#atom_sub_params,)*>;
                    fn len(&self) -> usize {
                        self.#first_entity_field.len()
                    }
                }
            }
        },
    );

    let truncate_impl = impl_simple_trait(
        ast,
        |ty_ident| quote! { #ty_ident: #crate_name::Truncate },
        |generics, ImplInfo { entity_field, .. }| {
            let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
            quote! {
                impl #impl_generics #crate_name::Truncate for #name #ty_generics #where_clause {
                    fn truncate(&mut self, len: usize) {
                        #(
                            self.#entity_field.truncate(len);
                        )*
                    }
                }
            }
        },
    );

    let clear_impl = impl_simple_trait(
        ast,
        |ty_ident| quote! { #ty_ident: #crate_name::Clear },
        |generics, ImplInfo { entity_field, .. }| {
            let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
            quote! {
                impl #impl_generics #crate_name::Clear for #name #ty_generics #where_clause {
                    fn clear(&mut self) {
                        #(
                            self.#entity_field.clear();
                        )*
                    }
                }
            }
        },
    );

    let into_owned_impl = impl_simple_trait(
        ast,
        |ty_ident| quote! { #ty_ident: #crate_name::IntoOwned },
        |generics,
         ImplInfo {
             entity_field,
             other_field,
             entity_type,
             ..
         }| {
            let sub_params =
                associated_type_params(parse_quote! { Owned }, &generics, &entity_type);
            let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
            quote! {
                impl #impl_generics #crate_name::IntoOwned for #name #ty_generics #where_clause {
                    type Owned = #name<#(#sub_params,)*>;
                    fn into_owned(self) -> Self::Owned {
                        #name {
                            #(
                                #entity_field: self.#entity_field.into_owned(),
                            )*
                            #(
                                #other_field: self.#other_field,
                            )*
                        }
                    }
                }
            }
        },
    );

    let into_owned_data_impl = impl_simple_trait(
        ast,
        |ty_ident| quote! { #ty_ident: #crate_name::IntoOwnedData },
        |generics,
         ImplInfo {
             entity_field,
             other_field,
             entity_type,
             ..
         }| {
            let sub_params =
                associated_type_params(parse_quote! { OwnedData }, &generics, &entity_type);
            let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
            quote! {
                impl #impl_generics #crate_name::IntoOwnedData for #name #ty_generics #where_clause {
                    type OwnedData = #name<#(#sub_params,)*>;
                    fn into_owned_data(self) -> Self::OwnedData {
                        #name {
                            #(
                                #entity_field: self.#entity_field.into_owned_data(),
                            )*
                            #(
                                #other_field: self.#other_field,
                            )*
                        }
                    }
                }
            }
        },
    );

    let remove_prefix_impl = impl_simple_trait(
        ast,
        |ty_ident| quote! { #ty_ident: #crate_name::RemovePrefix },
        |generics, ImplInfo { entity_field, .. }| {
            let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
            quote! {
                impl #impl_generics #crate_name::RemovePrefix for #name #ty_generics #where_clause {
                    fn remove_prefix(&mut self, n: usize) {
                        #(self.#entity_field.remove_prefix(n);)*
                    }
                }
            }
        },
    );

    let into_static_chunk_iterator_impl = {
        let impl_info = build_impl_info(ast);

        let mut generics = ast.generics.clone();
        let normal_where_clause = (*generics.make_where_clause()).clone();

        let where_clause = generics.make_where_clause();
        // Note: where clauses don't care about order so we can iterate over the entity_type set
        // directly.
        for ty_ident in &impl_info.entity_type {
            let bound_tokens = quote! { #ty_ident: #crate_name::IntoStaticChunkIterator<_FlatkN> };
            if !bound_tokens.is_empty() {
                where_clause
                    .predicates
                    .push(syn::parse2(bound_tokens).unwrap());
            }
        }

        let item_params =
            associated_type_params(parse_quote! { Item }, &generics, &impl_info.entity_type);
        let mut extended_generics = generics.clone();
        extended_generics.params.push(parse_quote! { _FlatkN });
        extended_generics
            .make_where_clause()
            .predicates
            .push(parse_quote! { _FlatkN: #crate_name::Unsigned });
        let (impl_generics, _, where_clause) = extended_generics.split_for_impl();
        let (normal_impl_generics, ty_generics, _) = generics.split_for_impl();

        // Note: the only reason we use a repeat () iterator, is to make the implementation
        // logic simpler in the body of the into_static_chunk_iter function.

        let mut zip_iter = quote! { std::iter::Repeat<()> };
        let mut tuple = quote! { () };
        let mut pattern = quote! { () };
        let mut zip_expr = quote! { std::iter::repeat(()) };
        let fields: Vec<_> = impl_info
            .field_type_attr
            .iter()
            .map(|(field, _, _)| field.clone())
            .collect();
        for (field, ty, is_entity) in impl_info.field_type_attr.iter() {
            tuple = quote! { (#tuple, #ty) };
            pattern = quote! { (#pattern, #field) };
            if *is_entity {
                zip_iter = quote! { std::iter::Zip<#zip_iter, <#ty as #crate_name::IntoStaticChunkIterator<_FlatkN>>::IterType> };
                zip_expr.extend(std::iter::once(quote! {
                    .zip(self.#field.into_static_chunk_iter())
                }));
                continue;
            }
            zip_iter = quote! { std::iter::Zip<#zip_iter, std::iter::Repeat<#ty>> };
            zip_expr.extend(std::iter::once(quote! {
                .zip(std::iter::repeat(self.#field))
            }));
        }

        quote! {
            impl #normal_impl_generics From<#tuple> for #name #ty_generics #normal_where_clause {
                fn from(#pattern: #tuple) -> Self {
                    #name {
                        #(#fields,)*
                    }
                }
            }

            impl #impl_generics #crate_name::IntoStaticChunkIterator<_FlatkN> for #name #ty_generics #where_clause {
                type Item = #name<#(#item_params,)*>;
                type IterType = #crate_name::StructIter<
                    #zip_iter,
                    Self::Item
                >;
                fn into_static_chunk_iter(self) -> Self::IterType {
                    #crate_name::StructIter::new(#zip_expr)
                }
            }
        }
    };

    let dummy_impl = impl_simple_trait(
        ast,
        |ty_ident| quote! { #ty_ident: #crate_name::Dummy },
        |generics,
         ImplInfo {
             entity_field,
             other_field,
             ..
         }| {
            let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
            quote! {
                impl #impl_generics #crate_name::Dummy for #name #ty_generics #where_clause {
                    unsafe fn dummy() -> Self {
                        #name {
                            #(
                                #entity_field: #crate_name::Dummy::dummy(),
                            )*
                            #(
                                #other_field: Default::default(),
                            )*
                        }
                    }
                }
            }
        },
    );

    let unichunkable_impl = impl_simple_trait(
        ast,
        |ty_ident| quote! { #ty_ident: #crate_name::UniChunkable<_FlatkN>},
        |generics, ImplInfo { entity_type, .. }| {
            let sub_params =
                associated_type_params(parse_quote! { Chunk }, &generics, &entity_type);

            let mut extended_generics = generics.clone();
            extended_generics.params.push(parse_quote! { _FlatkN });
            let (impl_generics, _, _) = extended_generics.split_for_impl();
            let (_, ty_generics, where_clause) = generics.split_for_impl();
            quote! {
                impl #impl_generics #crate_name::UniChunkable<_FlatkN> for #name #ty_generics #where_clause {
                    type Chunk = #name<#(#sub_params,)*>;
                }
            }
        },
    );

    let push_impl = impl_trait(
        ast,
        |ty_ident| {
            let alt_type = Ident::new(&format!("{}FlatkAlt", ty_ident), ty_ident.span());
            Some(parse_quote! { #alt_type })
        },
        |ty_ident, alt_ident| quote! { #ty_ident: #crate_name::Push<#alt_ident> },
        |generics, ext_generics, ImplInfo { entity_field, .. }, alt_type_params| {
            let (_, ty_generics, _) = generics.split_for_impl();
            let (impl_generics, _, where_clause) = ext_generics.split_for_impl();

            quote! {
                impl #impl_generics #crate_name::Push<#name<#(#alt_type_params,)*>> for #name #ty_generics #where_clause {
                    fn push(&mut self, elem: #name<#(#alt_type_params,)*>) {
                        let #name {
                            #(
                                #entity_field,
                            )*
                            ..
                        } = elem;

                        #(
                            self.#entity_field.push(#entity_field);
                        )*
                    }
                }
            }
        },
    );

    let storage_impls = impl_storage(ast);
    let get_impls = impl_get(ast);
    let isolate_impls = impl_isolate(ast);
    let view_impls = impl_view(ast);
    let split_impls = impl_split(ast);

    quote! {
        #value_type_impl
        #viewed_impl

        #set_impl
        #truncate_impl
        #clear_impl
        #into_owned_impl
        #into_owned_data_impl
        #into_static_chunk_iterator_impl
        #dummy_impl
        #remove_prefix_impl
        #unichunkable_impl
        #push_impl

        #storage_impls
        #get_impls
        #isolate_impls
        #view_impls
        #split_impls
    }
}
