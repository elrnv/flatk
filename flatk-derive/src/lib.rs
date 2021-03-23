extern crate proc_macro;
use proc_macro::TokenStream;

mod component;

use syn::{DeriveInput, Expr};

#[proc_macro_derive(Component, attributes(component))]
pub fn component(input: TokenStream) -> TokenStream {
    let input: DeriveInput = syn::parse(input).unwrap();

    let gen = component::impl_component(&input);

    gen.into()
}

/// Evaluate a numeric expression
fn eval_expr(expr: Expr) -> u64 {
    use syn::{BinOp, ExprBinary, ExprLit, ExprUnary, Lit, UnOp};
    match expr {
        Expr::Binary(ExprBinary {
            left, op, right, ..
        }) => {
            let l = eval_expr(*left);
            let r = eval_expr(*right);
            match op {
                BinOp::Add(_) => l + r,
                BinOp::Sub(_) => l - r,
                BinOp::Mul(_) => l * r,
                BinOp::Div(_) => l / r,
                BinOp::Rem(_) => l % r,
                BinOp::BitXor(_) => l ^ r,
                BinOp::BitAnd(_) => l & r,
                BinOp::BitOr(_) => l | r,
                BinOp::Shl(_) => l << r,
                BinOp::Shr(_) => l >> r,
                op => panic!("Unsupported operator type: {:?}", op),
            }
        }
        Expr::Unary(ExprUnary { op, expr, .. }) => {
            let a = eval_expr(*expr);
            match op {
                UnOp::Not(_) => !a,
                op => panic!("Unsupported operator type: {:?}", op),
            }
        }
        Expr::Lit(ExprLit {
            lit: Lit::Int(i), ..
        }) => i.base10_parse::<u64>().expect("Invalid integer literal"),
        _ => panic!("Unsupported expression type"),
    }
}

/// A simple macro constructing U# types given an expression.
///
/// For example: `U![3*3]` becomes `U9`.
#[allow(non_snake_case)]
#[proc_macro]
pub fn U(input: TokenStream) -> TokenStream {
    use proc_macro2::Span;
    use quote::quote;
    use syn::Ident;
    let expr: Expr = syn::parse(input).expect("Expected an expression");

    let num = eval_expr(expr);
    let ident = Ident::new(&format!("U{}", num), Span::call_site());
    (quote! {
        #ident
    })
    .into()
}
