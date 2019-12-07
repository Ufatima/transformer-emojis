# Creates a BERT friendly dataset from Twitter
import os
import re
import json
import argparse

import tweepy


def get_secrets(path="./secrets.json"):
    with open(path, "r") as f:
        return json.loads(f.read())


def get_emojis(path="./emoji_index.txt"):
    with open(path, "r") as f:
        return f.read().split(",")


def tokenize(string):
    return string.split("")


def extract_emojis(text):
    """ Extracts emojis from text. """
    return [e for e in emojis if e in text]


class EmojiStreamListener(tweepy.StreamListener):
    def __init__(self, args):
        super().__init__()
        self.idx = 0
        self.args = args

    def on_status(self, status):

        t = status._json
        text = t["extended_tweet"]["full_text"] if t["truncated"] else t["text"]

        # Remove tabs and spaces
        text = " ".join(text.split())

        # Remove links
        text = re.sub(r"http\S+", "", text)

        # Extract and remove emojis
        emos = extract_emojis(text)
        for e in emos:
            text = text.replace(e, "")

        # Remove non alphanumerics but leave spaces
        text = re.sub(r"([^\s\w]|_)+", "", text)

        for e in emos:
            emoji_idx = emojis.index(e)
            line = f"{self.idx}\t{emoji_idx}\ta\t{text}\n"
            self.idx += 1
            with open(os.path.join(self.args.out_dir, t["lang"], "all.tsv"), "a+") as f:
                f.write(line)

            if self.idx % self.args.progress_every == 0:
                print(f"Saved {self.idx} tweets so far")


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Create tweet emoji dataset")
    parser.add_argument(
        "--langs",
        type=str,
        help="Comma separaed list of BCP 47 language identifiers",
        default="en",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Where to save the dataset",
        default="./data/tweet_emoji_dataset",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        help="Print the progress every n tweet",
        default=100,
    )

    args = parser.parse_args()
    args.langs = args.langs.split(",")

    secrets = get_secrets()
    emojis = get_emojis()

    auth = tweepy.OAuthHandler(secrets["consumer_key"], secrets["consumer_secret"])
    auth.set_access_token(secrets["access_token_key"], secrets["access_token_secret"])
    api = tweepy.API(auth)

    emojiStreamListener = EmojiStreamListener(args)
    stream = tweepy.Stream(auth=api.auth, listener=emojiStreamListener)

    for lang in args.langs:
        lang_path = os.path.join(args.out_dir, lang)
        if not os.path.exists(lang_path):
            os.makedirs(lang_path)

    print("Commencing streaming, exit with CTRL+C")
    try:
        stream.filter(track=emojis, languages=args.langs)
    except KeyboardInterrupt:
        print("\nCTRL+C detected, bye bye :)")

