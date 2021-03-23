use std::collections::{BTreeSet, HashSet};

use lazy_static::lazy_static;
use proc_macro2::{Span, TokenStream};
use proc_macro_crate::*;
use quote::quote;
use syn::*;

//TODO: Add another impl trait for CloneWithStorage

lazy_static! {
    static ref CRATE_NAME: String = {
        // Try to find the crate name in Cargo.toml
        if let Ok(found_name) = crate_name("flatk") {
            match found_name {
                FoundCrate::Itself => String::from("flatk"),
                FoundCrate::Name(name) => name
            }
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
    component_type: &BTreeSet<Ident>,
) -> Vec<Type> {
    // Populate parameters for the associated type
    generics
        .type_params()
        .map(|TypeParam { ident, .. }| {
            if component_type.contains(ident) {
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
             component_field,
             other_field,
             ..
         }| {
            let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
            quote! {
                impl #impl_generics #crate_name::SplitAt for #name #ty_generics #where_clause {
                    fn split_at(self, mid: usize) -> (Self, Self) {
                        let #name {
                            #(
                                #component_field,
                            )*
                            #(
                                #other_field,
                            )*
                        } = self;

                        #(
                            let #component_field = #component_field.split_at(mid);
                        )*
                        (
                            #name {
                                #(
                                    #component_field: #component_field.0,
                                )*
                                #(
                                    #other_field: #other_field.clone(),
                                )*
                            },
                            #name {
                                #(
                                    #component_field: #component_field.1,
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
             component_field,
             other_field,
             ..
         }| {
            let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
            quote! {
                impl #impl_generics #crate_name::SplitOff for #name #ty_generics #where_clause {
                    fn split_off(&mut self, mid: usize) -> Self {
                        let #name {
                            #(
                                ref mut #component_field,
                            )*
                            #(
                                #other_field,
                            )*
                        } = *self;

                        #(
                            let #component_field = #component_field.split_off(mid);
                        )*
                        #name {
                            #(
                                #component_field,
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
             component_field,
             other_field,
             component_type,
             ..
         },
         _| {
            let sub_params =
                associated_type_params(parse_quote! { Prefix }, &generics, &component_type);
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
                                #component_field,
                            )*
                            #(
                                #other_field,
                            )*
                        } = self;

                        #(
                            let #component_field = #component_field.split_prefix()?;
                        )*

                        Some((
                            #name {
                                #(
                                    #component_field: #component_field.0,
                                )*
                                #(
                                    #other_field: #other_field.clone(),
                                )*
                            },
                            #name {
                                #(
                                    #component_field: #component_field.1,
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
             component_field,
             other_field,
             component_type,
             ..
         }| {
            let sub_params =
                associated_type_params(parse_quote! { First }, &generics, &component_type);
            let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
            quote! {
                impl #impl_generics #crate_name::SplitFirst for #name #ty_generics #where_clause {
                    type First = #name<#(#sub_params,)*>;
                    fn split_first(self) -> Option<(Self::First, Self)> {
                        let #name {
                            #(
                                #component_field,
                            )*
                            #(
                                #other_field,
                            )*
                        } = self;

                        #(
                            let #component_field = #component_field.split_first()?;
                        )*

                        Some((
                            #name {
                                #(
                                    #component_field: #component_field.0,
                                )*
                                #(
                                    #other_field: #other_field.clone(),
                                )*
                            },
                            #name {
                                #(
                                    #component_field: #component_field.1,
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
             component_field,
             other_field,
             component_type,
             ..
         }| {
            let sub_params =
                associated_type_params(parse_quote! { StorageType }, &generics, &component_type);
            let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
            quote! {
                impl #impl_generics #crate_name::IntoStorage for #name #ty_generics #where_clause {
                    type StorageType = #name<#(#sub_params,)*>;
                    fn into_storage(self) -> Self::StorageType {
                        #name {
                            #(
                                #component_field: self.#component_field.into_storage(),
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
             component_field,
             other_field,
             component_type,
             ..
         },
         alt_type_params| {
            let sub_params =
                associated_type_params(parse_quote! { Output }, &generics, &component_type);
            let (_, ty_generics, _) = generics.split_for_impl();
            let (impl_generics, _, where_clause) = ext_generics.split_for_impl();

            quote! {
                impl #impl_generics #crate_name::StorageInto<#name<#(#alt_type_params,)*>> for #name #ty_generics #where_clause {
                    type Output = #name<#(#sub_params,)*>;
                    #[inline]
                    fn storage_into(self) -> Self::Output {
                        #name {
                            #(
                                #component_field: self.#component_field.storage_into(),
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
        |_| quote! {},
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
        |_| quote! {},
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
             component_field,
             other_field,
             component_type,
             ..
         }| {
            let sub_params =
                associated_type_params(parse_quote! { Output }, &generics, &component_type);
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
                                #component_field: this.#component_field.get(self)?,
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
        |ty_ident| quote! { #ty_ident: #crate_name::Get<'flatk_get, core::ops::Range<usize>> },
        |generics,
         ImplInfo {
             component_field,
             other_field,
             component_type,
             ..
         }| {
            let sub_params =
                associated_type_params(parse_quote! { Output }, &generics, &component_type);
            let mut extended_generics = generics.clone();
            extended_generics.params.push(parse_quote! { 'flatk_get});
            let (impl_generics, _, _) = extended_generics.split_for_impl();
            let (_, ty_generics, where_clause) = generics.split_for_impl();
            quote! {
                impl #impl_generics #crate_name::GetIndex<'flatk_get, #name #ty_generics> for ::core::ops::Range<usize>  #where_clause {
                    type Output = #name<#(#sub_params,)*>;
                    fn get(self, this: &#name #ty_generics) -> Option<Self::Output> {

                        Some(#name {
                            #(
                                #component_field: this.#component_field.get(self.clone())?,
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
        |ty_ident| quote! { #ty_ident: #crate_name::Get<'flatk_get, #crate_name::StaticRange<_FlatkN>> },
        |mut generics,
         ImplInfo {
             component_field,
             other_field,
             component_type,
             ..
         }| {
            generics
                .make_where_clause()
                .predicates
                .push(parse_quote! { _FlatkN: #crate_name::Unsigned + Copy });
            let sub_params =
                associated_type_params(parse_quote! { Output }, &generics, &component_type);
            let mut extended_generics = generics.clone();
            extended_generics.params.push(parse_quote! { _FlatkN });
            extended_generics.params.push(parse_quote! { 'flatk_get });
            let (impl_generics, _, _) = extended_generics.split_for_impl();
            let (_, ty_generics, where_clause) = generics.split_for_impl();
            quote! {
                impl #impl_generics #crate_name::GetIndex<'flatk_get, #name #ty_generics> for #crate_name::StaticRange<_FlatkN>  #where_clause {
                    type Output = #name<#(#sub_params,)*>;
                    #[inline]
                    fn get(self, this: &#name #ty_generics) -> Option<Self::Output> {

                        Some(#name {
                            #(
                                #component_field: this.#component_field.get(self)?,
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
             component_field,
             other_field,
             component_type,
             ..
         }| {
            let sub_params =
                associated_type_params(parse_quote! { Output }, &generics, &component_type);
            let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
            quote! {
                impl #impl_generics #crate_name::IsolateIndex<#name #ty_generics> for usize  #where_clause {
                    type Output = #name<#(#sub_params,)*>;
                    #[inline]
                    unsafe fn isolate_unchecked(self, this: #name #ty_generics) -> Self::Output {
                        #name {
                            #(
                                #component_field: #crate_name::Isolate::isolate_unchecked(this.#component_field, self),
                            )*
                            #(
                                #other_field: this.#other_field,
                            )*
                        }
                    }
                    #[inline]
                    fn try_isolate(self, this: #name #ty_generics) -> Option<Self::Output> {
                        Some(#name {
                            #(
                                #component_field: #crate_name::Isolate::try_isolate(this.#component_field, self)?,
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
        |ty_ident| quote! { #ty_ident: #crate_name::Isolate<core::ops::Range<usize>> },
        |generics,
         ImplInfo {
             component_field,
             other_field,
             component_type,
             ..
         }| {
            let sub_params =
                associated_type_params(parse_quote! { Output }, &generics, &component_type);
            let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
            quote! {
                impl #impl_generics #crate_name::IsolateIndex<#name #ty_generics> for ::core::ops::Range<usize>  #where_clause {
                    type Output = #name<#(#sub_params,)*>;
                    #[inline]
                    unsafe fn isolate_unchecked(self, this: #name #ty_generics) -> Self::Output {
                        #name {
                            #(
                                #component_field: #crate_name::Isolate::isolate_unchecked(this.#component_field, self.clone()),
                            )*
                            #(
                                #other_field: this.#other_field,
                            )*
                        }
                    }
                    #[inline]
                    fn try_isolate(self, this: #name #ty_generics) -> Option<Self::Output> {
                        Some(#name {
                            #(
                                #component_field: #crate_name::Isolate::try_isolate(this.#component_field, self.clone())?,
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

    let isolate_index_for_static_range_impl = impl_simple_trait(
        ast,
        |ty_ident| quote! { #ty_ident: #crate_name::Isolate<#crate_name::StaticRange<_FlatkN>> },
        |mut generics,
         ImplInfo {
             component_field,
             other_field,
             component_type,
             ..
         }| {
            generics
                .make_where_clause()
                .predicates
                .push(parse_quote! { _FlatkN: #crate_name::Unsigned });
            let sub_params =
                associated_type_params(parse_quote! { Output }, &generics, &component_type);
            let mut extended_generics = generics.clone();
            extended_generics.params.push(parse_quote! { _FlatkN });
            let (impl_generics, _, _) = extended_generics.split_for_impl();
            let (_, ty_generics, where_clause) = generics.split_for_impl();
            quote! {
                impl #impl_generics #crate_name::IsolateIndex<#name #ty_generics> for #crate_name::StaticRange<_FlatkN>  #where_clause {
                    type Output = #name<#(#sub_params,)*>;
                    #[inline]
                    unsafe fn isolate_unchecked(self, this: #name #ty_generics) -> Self::Output {
                        #name {
                            #(
                                #component_field: #crate_name::Isolate::isolate_unchecked(this.#component_field, self.clone()),
                            )*
                            #(
                                #other_field: this.#other_field,
                            )*
                        }
                    }
                    #[inline]
                    fn try_isolate(self, this: #name #ty_generics) -> Option<Self::Output> {
                        Some(#name {
                            #(
                                #component_field: #crate_name::Isolate::try_isolate(this.#component_field, self.clone())?,
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
        #isolate_index_for_static_range_impl
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
             component_field,
             other_field,
             component_type,
             ..
         }| {
            let sub_params =
                associated_type_params(parse_quote! { Type }, &generics, &component_type);
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
                                #component_field: self.#component_field.view(),
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
             component_field,
             other_field,
             component_type,
             ..
         }| {
            let sub_params =
                associated_type_params(parse_quote! { Type }, &generics, &component_type);

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
                                #component_field: self.#component_field.view_mut(),
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

fn impl_iter(ast: &DeriveInput) -> TokenStream {
    let name = &ast.ident;

    let crate_name = crate_name_ident();

    // Strictly speaking this is not a trait impl, but the same mechanism works here too.
    let iter_impl = impl_simple_trait(
        ast,
        |_| quote! {},
        |generics, ImplInfo { component_type, .. }| {
            let predicates = component_type
                .iter()
                .flat_map(|ty_ident| {
                    let main_pred: WherePredicate = parse_quote! { #ty_ident: View<'s> };
                    let iter_pred: WherePredicate =
                        parse_quote! { <#ty_ident as View<'s>>::Type: core::iter::IntoIterator };
                    std::iter::once(main_pred).chain(std::iter::once(iter_pred))
                })
                .collect();

            let iter_where_clause = WhereClause {
                where_token: parse_quote! { where },
                predicates,
            };

            let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

            quote! {
                impl #impl_generics #name #ty_generics #where_clause {
                    pub fn iter<'s>(&'s self) -> <<Self as View<'s>>::Type as IntoIterator>::IntoIter
                        #iter_where_clause
                    {
                        self.view().into_iter()
                    }
                }
            }
        },
    );

    let iter_mut_impl = impl_simple_trait(
        ast,
        |_| quote! {},
        |generics, ImplInfo { component_type, .. }| {
            let predicates = component_type
                .iter()
                .flat_map(|ty_ident| {
                    let main_pred: WherePredicate = parse_quote! { #ty_ident: ViewMut<'s> };
                    let iter_pred: WherePredicate =
                        parse_quote! { <#ty_ident as ViewMut<'s>>::Type: core::iter::IntoIterator };
                    std::iter::once(main_pred).chain(std::iter::once(iter_pred))
                })
                .collect();

            let iter_where_clause = WhereClause {
                where_token: parse_quote! { where },
                predicates,
            };

            let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

            quote! {
                impl #impl_generics #name #ty_generics #where_clause {
                    pub fn iter_mut<'s>(&'s mut self) -> <<Self as ViewMut<'s>>::Type as IntoIterator>::IntoIter
                        #iter_where_clause
                    {
                        self.view_mut().into_iter()
                    }
                }
            }
        },
    );

    let into_iter_impl = impl_trait(
        ast,
        |ty_ident| {
            let alt_type = Ident::new(&format!("{}FlatkAlt", ty_ident), ty_ident.span());
            Some(parse_quote! { #alt_type })
        },
        |ty_ident, alt_param| {
            if let Some(GenericParam::Type(alt_type_param)) = alt_param {
                let alt_ident = &alt_type_param.ident;
                quote! { #ty_ident: core::iter::IntoIterator<Item = #alt_ident> }
            } else {
                quote! {}
            }
        },
        |generics,
         ext_generics,
         ImplInfo {
             component_type,
             field_type_attr,
             ..
         },
         _| {
            let item_params =
                associated_type_params(parse_quote! { Item }, &generics, &component_type);

            let (_, ty_generics, _) = generics.split_for_impl();
            let (impl_generics, _, where_clause) = ext_generics.split_for_impl();

            let mut zip_iter = quote! { core::iter::Repeat<()> };
            let mut tuple = quote! { () };
            let mut zip_expr = quote! { core::iter::repeat(()) };
            for (field, ty, is_component) in field_type_attr.iter() {
                tuple = quote! { (#tuple, #ty) };
                if *is_component {
                    zip_iter = quote! { core::iter::Zip<#zip_iter, <#ty as core::iter::IntoIterator>::IntoIter> };
                    zip_expr.extend(core::iter::once(quote! {
                        .zip(self.#field.into_iter())
                    }));
                    continue;
                }
                zip_iter = quote! { core::iter::Zip<#zip_iter, core::iter::Repeat<#ty>> };
                zip_expr.extend(core::iter::once(quote! {
                    .zip(core::iter::repeat(self.#field))
                }));
            }

            quote! {
                impl #impl_generics core::iter::IntoIterator for #name #ty_generics #where_clause {
                    type Item = #name<#(#item_params,)*>;
                    type IntoIter = #crate_name::StructIter< #zip_iter, Self::Item >;

                    fn into_iter(self) -> Self::IntoIter {
                        #crate_name::StructIter::new(#zip_expr)
                    }
                }
            }
        },
    );

    quote! {
        #iter_impl
        #iter_mut_impl
        #into_iter_impl
    }
}

/// A selection of information useful for implementing component traits.
#[derive(Clone, Debug, Default)]
struct ImplInfo {
    pub field_type_attr: Vec<(Ident, Type, bool)>,
    pub component_field: BTreeSet<Ident>,
    pub other_field: BTreeSet<Ident>,
    pub component_type: BTreeSet<Ident>,
}

fn build_impl_info(ast: &DeriveInput) -> ImplInfo {
    let type_params: HashSet<_> = ast
        .generics
        .type_params()
        .map(|t| t.ident.clone())
        .collect();

    let mut field_type_attr = Vec::new();
    let mut component_field = BTreeSet::new();
    let mut other_field = BTreeSet::new();
    let mut component_type = BTreeSet::new();

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
                        component_field.insert(ident.clone().unwrap());
                        component_type.insert(ty_ident.clone());
                        field_type_attr.push((ident.clone().unwrap(), ty.clone(), true));
                        continue;
                    }
                }
                // Or we look for a component attribute for passthrough components.
                if attrs.contains(&parse_quote! { #[component] }) {
                    let last = path
                        .segments
                        .last()
                        .expect("Component field must be a named type");
                    if let PathArguments::AngleBracketed(all_args) = &last.arguments {
                        // All type parameters are assumed to be Component parameters,
                        // Since it's impossible to tell which parameters are significant, without
                        // also knowing the Component impls of this type.

                        assert!(
                            !all_args.args.is_empty(),
                            "Component fields must have at least one type parameter"
                        );
                        for arg in all_args.args.iter() {
                            if let GenericArgument::Type(Type::Path(TypePath {
                                qself: None,
                                path,
                            })) = arg
                            {
                                if let Some(ty_ident) = path.get_ident() {
                                    component_type.insert(ty_ident.clone());
                                }
                            }
                        }
                        component_field.insert(ident.clone().unwrap());
                        field_type_attr.push((ident.clone().unwrap(), ty.clone(), true));
                        continue;
                    } else {
                        panic!("Component fields must have at least one type parameter");
                    }
                }
            }
            field_type_attr.push((ident.clone().unwrap(), ty.clone(), false));
            other_field.insert(ident.clone().unwrap());
        }
    }

    ImplInfo {
        field_type_attr,
        component_field,
        other_field,
        component_type,
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
        if impl_info.component_type.contains(ty_ident) {
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

/// Implement a trait without additional type parameters.
fn impl_simple_trait(
    ast: &DeriveInput,
    mut generate_bound: impl FnMut(&Ident) -> TokenStream,
    generate_impl: impl FnOnce(Generics, ImplInfo) -> TokenStream,
) -> TokenStream {
    let impl_info = build_impl_info(&ast);

    let mut generics = ast.generics.clone();

    let where_clause = generics.make_where_clause();
    // Note: where clauses don't care about order so we can iterate over the component_type set
    // directly.
    for ty_ident in &impl_info.component_type {
        let bound_tokens = generate_bound(ty_ident);
        if !bound_tokens.is_empty() {
            where_clause
                .predicates
                .push(syn::parse2(bound_tokens).unwrap());
        }
    }

    generate_impl(generics, impl_info)
}

pub(crate) fn impl_component(ast: &DeriveInput) -> TokenStream {
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
             component_field,
             component_type,
             ..
         }| {
            let first_component_field = component_field
                .iter()
                .next()
                .expect("Component types require at least one component");
            let elem_sub_params =
                associated_type_params(parse_quote! { Elem }, &generics, &component_type);
            let atom_sub_params =
                associated_type_params(parse_quote! { Atom }, &generics, &component_type);

            let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
            quote! {
                impl #impl_generics #crate_name::Set for #name #ty_generics #where_clause {
                    type Elem = #name<#(#elem_sub_params,)*>;
                    type Atom = #name<#(#atom_sub_params,)*>;
                    fn len(&self) -> usize {
                        self.#first_component_field.len()
                    }
                }
            }
        },
    );

    let truncate_impl = impl_simple_trait(
        ast,
        |ty_ident| quote! { #ty_ident: #crate_name::Truncate },
        |generics,
         ImplInfo {
             component_field, ..
         }| {
            let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
            quote! {
                impl #impl_generics #crate_name::Truncate for #name #ty_generics #where_clause {
                    fn truncate(&mut self, len: usize) {
                        #(
                            self.#component_field.truncate(len);
                        )*
                    }
                }
            }
        },
    );

    let clear_impl = impl_simple_trait(
        ast,
        |ty_ident| quote! { #ty_ident: #crate_name::Clear },
        |generics,
         ImplInfo {
             component_field, ..
         }| {
            let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
            quote! {
                impl #impl_generics #crate_name::Clear for #name #ty_generics #where_clause {
                    fn clear(&mut self) {
                        #(
                            self.#component_field.clear();
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
             component_field,
             other_field,
             component_type,
             ..
         }| {
            let sub_params =
                associated_type_params(parse_quote! { Owned }, &generics, &component_type);
            let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
            quote! {
                impl #impl_generics #crate_name::IntoOwned for #name #ty_generics #where_clause {
                    type Owned = #name<#(#sub_params,)*>;
                    fn into_owned(self) -> Self::Owned {
                        #name {
                            #(
                                #component_field: self.#component_field.into_owned(),
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
             component_field,
             other_field,
             component_type,
             ..
         }| {
            let sub_params =
                associated_type_params(parse_quote! { OwnedData }, &generics, &component_type);
            let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
            quote! {
                impl #impl_generics #crate_name::IntoOwnedData for #name #ty_generics #where_clause {
                    type OwnedData = #name<#(#sub_params,)*>;
                    fn into_owned_data(self) -> Self::OwnedData {
                        #name {
                            #(
                                #component_field: self.#component_field.into_owned_data(),
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
        |generics,
         ImplInfo {
             component_field, ..
         }| {
            let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
            quote! {
                impl #impl_generics #crate_name::RemovePrefix for #name #ty_generics #where_clause {
                    fn remove_prefix(&mut self, n: usize) {
                        #(self.#component_field.remove_prefix(n);)*
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
        // Note: where clauses don't care about order so we can iterate over the component_type set
        // directly.
        for ty_ident in &impl_info.component_type {
            let bound_tokens = quote! { #ty_ident: #crate_name::IntoStaticChunkIterator<_FlatkN> };
            if !bound_tokens.is_empty() {
                where_clause
                    .predicates
                    .push(syn::parse2(bound_tokens).unwrap());
            }
        }

        let item_params =
            associated_type_params(parse_quote! { Item }, &generics, &impl_info.component_type);
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

        let mut zip_iter = quote! { core::iter::Repeat<()> };
        let mut tuple = quote! { () };
        let mut pattern = quote! { () };
        let mut zip_expr = quote! { core::iter::repeat(()) };
        let fields: Vec<_> = impl_info
            .field_type_attr
            .iter()
            .map(|(field, _, _)| field.clone())
            .collect();
        for (field, ty, is_component) in impl_info.field_type_attr.iter() {
            tuple = quote! { (#tuple, #ty) };
            pattern = quote! { (#pattern, #field) };
            if *is_component {
                zip_iter = quote! { core::iter::Zip<#zip_iter, <#ty as #crate_name::IntoStaticChunkIterator<_FlatkN>>::IterType> };
                zip_expr.extend(core::iter::once(quote! {
                    .zip(self.#field.into_static_chunk_iter())
                }));
                continue;
            }
            zip_iter = quote! { core::iter::Zip<#zip_iter, core::iter::Repeat<#ty>> };
            zip_expr.extend(core::iter::once(quote! {
                .zip(core::iter::repeat(self.#field))
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
             component_field,
             other_field,
             ..
         }| {
            let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
            quote! {
                impl #impl_generics #crate_name::Dummy for #name #ty_generics #where_clause {
                    unsafe fn dummy() -> Self {
                        #name {
                            #(
                                #component_field: #crate_name::Dummy::dummy(),
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
        |generics, ImplInfo { component_type, .. }| {
            let sub_params =
                associated_type_params(parse_quote! { Chunk }, &generics, &component_type);

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

    let push_chunk_impl = impl_simple_trait(
        ast,
        |ty_ident| quote! { #ty_ident: #crate_name::PushChunk<_FlatkN> },
        |generics,
         ImplInfo {
             component_field, ..
         }| {
            let mut extended_generics = generics.clone();
            extended_generics.params.push(parse_quote! { _FlatkN });
            let (impl_generics, _, _) = extended_generics.split_for_impl();
            let (_, ty_generics, where_clause) = generics.split_for_impl();

            quote! {
                impl #impl_generics #crate_name::PushChunk<_FlatkN> for #name #ty_generics #where_clause {
                    fn push_chunk(&mut self, chunk: Self::Chunk) {
                        #(
                            self.#component_field.push_chunk(chunk.#component_field);
                        )*
                        // Other fields are ignored.
                    }
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
        |generics,
         ext_generics,
         ImplInfo {
             component_field, ..
         },
         alt_type_params| {
            let (_, ty_generics, _) = generics.split_for_impl();
            let (impl_generics, _, where_clause) = ext_generics.split_for_impl();

            quote! {
                impl #impl_generics #crate_name::Push<#name<#(#alt_type_params,)*>> for #name #ty_generics #where_clause {
                    fn push(&mut self, elem: #name<#(#alt_type_params,)*>) {
                        let #name {
                            #(
                                #component_field,
                            )*
                            ..
                        } = elem;

                        #(
                            self.#component_field.push(#component_field);
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
    let iter_impls = impl_iter(ast);

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
        #push_chunk_impl

        #storage_impls
        #get_impls
        #isolate_impls
        #view_impls
        #split_impls

        #iter_impls
    }
}
