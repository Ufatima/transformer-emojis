# Creates a BERT friendly dataset from Twitter
import os
import re
import json
import argparse

import tweepy


def get_secrets(path):
    with open(path, "r") as f:
        return json.loads(f.read())


def get_emojis(path):
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

        # Extract and remove emojis
        emos = extract_emojis(text)
        for e in emos:
            text = text.replace(e, "")

        # Remove tabs and spaces
        text = " ".join(text.split())

        # Remove RT at the beginnin
        text = re.sub(r"^RT ", "", text)

        # Remove links, user links, and hashtags
        text = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", text)

        # # Remove hashtags
        # text = re.sub(r"/^#\w+$/", "", text)

        # Remove non alphanumerics but leave spaces
        text = re.sub(r"([^\s\w]|_)+", "", text)

        # Remove trailing whitespace
        text = text.strip()

        for i, e in enumerate(emos):
            emoji_idx = emojis.index(e)
            line_id = t["id_str"] + str(i)
            line = f"{line_id}\t{emoji_idx}\ta\t{text}\n"
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
    parser.add_argument(
        "--secrets",
        type=str,
        help="Path to the secrets.json file holding your Twitter app credentials",
        default="./secrets.json",
    )
    parser.add_argument(
        "--emojis",
        type=str,
        help="Path to the file containing a list of emojis to track",
        default="./emoji_index.txt",
    )

    args = parser.parse_args()
    args.langs = args.langs.split(",")

    print(f"Collecting tweets with languages {' '.join(args.langs)}")

    secrets = get_secrets(args.secrets)
    emojis = get_emojis(args.emojis)

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

