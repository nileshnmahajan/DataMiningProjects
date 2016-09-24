
# df_seg = df_train['Structure'].str[1:-1].str.split(' ').apply(pd.Series)
# # print df_seg.convert_objects(convert_numeric=True).dtypes
#
#
# # print type(df_seg)
#
# result = pd.concat([df_train, df_seg], axis=1)
#
# # print result
#
# # find_length = lambda x: len(x)
# #
# # segments = lambda x: len(x.split())
# #
# # df_train['Length'] = df_train['Structure'].apply(find_length)
# # df_train['Segments'] = df_train['Structure'].apply(segments)
#
# # print df_train['Length']
# # print df_train['Segments']
#
# # print df_train['Segments'].sort_values(axis=0, ascending=True, inplace=False, na_position='last')
#
# # print df_train['Segments'].mode()
#
# selector = VarianceThreshold()
# selector.fit_transform(result)