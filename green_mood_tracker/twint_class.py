import twint
import pandas as pd
import os


class TWINT():

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.username = kwargs.get('username', False)
        self.keywords = kwargs.get('keywords', False)
        self.since = kwargs.get('since', False)
        self.cities = kwargs.get('cities', False)
        self.file_path = kwargs.get('file_path', False)
        self.limit = kwargs.get('limit', False)
        self.overwrite = kwargs.get('overwrite', True)

    def search(self):
        c = twint.Config()

        c.Limit = self.limit
        if self.username:
            c.Username = self.username
        c.Hide_output = True
        c.Search = self.keywords
        c.Since = self.since
        c.Pandas = True
        twint.run.Search(c)

        Tweets_df = twint.storage.panda.Tweets_df
        Tweets_df = Tweets_df[Tweets_df['language'] == 'en']
        return Tweets_df

    def csv(self):
        if self.overwrite:
            os.system(f'rm -f {self.file_path} 2> /dev/null')

        c = twint.Config()

        if self.username:
            c.Username = self.username
        c.Limit = self.limit
        c.Hide_output = True
        c.Store_csv = True
        c.Output = './' + self.file_path
        c.Search = self.keywords
        c.Since = self.since

        twint.run.Search(c)

    def city_df(self):
        self.cities = sorted(set(self.cities))
        list_df = []

        for city in self.cities:
            c = twint.Config()
            c.Limit = self.limit
            c.Search = self.keywords
            c.Since = self.since
            c.Hide_output = True
            c.Near = city
            c.Pandas = True
            twint.run.Search(c)

            Tweets_df = twint.storage.panda.Tweets_df
            Tweets_df = Tweets_df[Tweets_df['language'] == 'en']

            list_df.append(Tweets_df)

        df = pd.concat(list_df, ignore_index=True)
        return df

    def city_csv(self):
        if self.overwrite:
            os.system(f'rm -f {self.file_path} 2> /dev/null')

        self.cities = sorted(set(self.cities))

        for city in self.cities:
            c = twint.Config()
            c.Limit = self.limit
            c.Search = self.keywords
            c.Store_csv = True
            c.Output = './' + self.file_path
            c.Since = self.since
            c.Hide_output = True
            c.Near = city
            twint.run.Search(c)

        csv = pd.read_csv('./' + self.file_path)
        csv = csv[csv['language'] == 'en']
        csv.to_csv('./' + self.file_path, index=False)


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
