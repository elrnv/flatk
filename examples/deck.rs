use flatk::Get;
use flatk::Subset;
use flatk::View;

fn main() {
    let rank = vec![
        "Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King",
    ];
    let suit = vec!["Clubs", "Diamonds", "Hearts", "Spades"];

    let deck: (Vec<_>, Vec<_>) = (
        rank.into_iter().cycle().take(52).collect(),
        suit.into_iter().cycle().take(52).collect(),
    );

    let hand = Subset::from_indices(vec![4, 19, 23, 1, 0, 5], deck);
    let hand_view = hand.view();
    assert_eq!(hand_view.at(0), (&"Ace", &"Clubs"));
    assert_eq!(hand_view.at(1), (&"2", &"Diamonds"));
    assert_eq!(hand_view.at(2), (&"5", &"Clubs"));
    assert_eq!(hand_view.at(3), (&"6", &"Diamonds"));
    assert_eq!(hand_view.at(4), (&"7", &"Spades"));
    assert_eq!(hand_view.at(5), (&"Jack", &"Spades"));
}
