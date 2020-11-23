# TWINT class

import twint


class TWINT():

    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self.username = kwargs.get("username", False)
        self.search = kwargs.get("search", False)
        self.store_csv = kwargs.get("store_csv", False)
        self.limit = kwargs.get("limit", False)

    def save(self, file_path):
        c = twint.Config()

        c.Username = self.username
        c.Lang = 'en'
        c.Search = self.search

        c.Store_csv = self.store_csv
        c.Output = file_path

        twint.run.Search(c)

    def dataframe(self):

        c = twint.Config()

        c.Limit = self.limit
        c.Username = self.username
        # c.Lang = 'en' To check out
        c.Search = self.search

        c.Pandas = True
        twint.run.Search(c)

        Tweets_df = twint.storage.panda.Tweets_df

        Tweets_df = Tweets_df[Tweets_df['language'] == 'en']

        return Tweets_df


def main():

    kwargs = dict(
        username='realDonaldTrump',
        search='french',
        store_csv=True,
        limit=2
    )

    t = TWINT(**kwargs)

    t.save(file_path='french.csv')


if __name__ == '__main__':
    main()
