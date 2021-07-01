# import tensorflow as tf
# from ..fix import fix_spaces, fix_tokens, fix_sents, fix_predictions
#
#
# class TestFixSpaces(tf.test.TestCase):
#     def test_ok(self):
#         expected = [False, True, False, True, False, True, False]
#         result = fix_spaces(
#             ['ab', ' ', 'cd', '\t', 'ef', '\n', 'gh'],
#             [False, True, False, True, False, True, False])
#         self.assertListEqual(expected, result)
#
#     def test_punkt(self):
#         expected = [False, True, False, True, False, True, False]
#         result = fix_spaces(
#             ['ab', ' ', 'cd', '\t', 'ef', '\n', '=:)'],
#             [False, True, False, True, False, True, True])
#         self.assertListEqual(expected, result)
#
#     def test_token(self):
#         expected = [False, False, False, True, False, True, False]
#         result = fix_spaces(
#             ['ab', ' c', 'd', '\t', 'ef', '\n', 'gh'],
#             [False, True, False, True, False, True, False])
#         self.assertListEqual(expected, result)
#
#     def test_space(self):
#         expected = [False, True, False, True, False, True, False]
#         result = fix_spaces(
#             ['ab', ' ', 'cd', '\t', 'ef', '\n', 'gh'],
#             [False, False, False, True, False, True, False])
#         self.assertListEqual(expected, result)
#
#
# class TestFixTokens(tf.test.TestCase):
#     def test_ok(self):
#         expected = [True, True, True, False, True, False, True]
#         result = fix_tokens(  # a_bc__d
#             [False, True, False, False, True, True, False],
#             [True, True, True, False, True, False, True])
#         self.assertListEqual(expected, result)
#
#     def test_inside(self):
#         expected = [True, True, True, False, True, False, True]
#         result = fix_tokens(  # a_bc__d
#             [False, True, False, False, True, True, False],
#             [False, True, True, False, True, True, True])
#         self.assertListEqual(expected, result)
#
#     def test_border(self):
#         expected = [True, True, True, False, True, False, True]
#         result = fix_tokens(  # a_bc__d
#             [False, True, False, False, True, True, False],
#             [False, False, True, False, True, False, False]
#         )
#         self.assertListEqual(expected, result)
#
#
# class TestFixSents(tf.test.TestCase):
#     def test_ok(self):
#         expected = [True, False, True, False, False, False, False]
#         result = fix_sents(  # a_ bc__d
#             [False, True, False, False, True, True, False],
#             [True, True, True, False, True, False, True],
#             [True, False, True, False, False, False, False])
#         self.assertListEqual(expected, result)
#
#     def test_space(self):
#         expected = [True, False, True, False, False, False, False]
#         result = fix_sents(  # a_ bc__d
#             [False, True, False, False, True, True, False],
#             [True, True, True, False, True, False, True],
#             [False, True, False, False, False, False, False])
#         self.assertListEqual(expected, result)
#
#     def test_inside(self):
#         expected = [True, False, True, False, False, False, False, False]
#         result = fix_sents(  # a_ bcd__e
#             [False, True, False, False, False, True, True, False],
#             [True, True, True, False, False, True, False, True],
#             [True, False, False, False, True, False, False, False])
#         self.assertListEqual(expected, result)
#
#
# class TestFixPredictions(tf.test.TestCase):
#     def test_ok(self):
#         expected = (
#             ['a', ' ', 'b', 'c', ' ', '\t', 'd'],
#             [False, True, False, False, True, True, False],
#             [True, True, True, False, True, False, True],
#             [True, False, True, False, False, False, False])
#         result = fix_predictions(
#             ['a', ' ', 'b', 'c', ' ', '\t', 'd'],
#             [False, True, False, False, True, True, False],
#             [True, True, True, False, True, False, True],
#             [True, False, True, False, False, False, False])
#         self.assertListEqual(expected[0], result[0])
#         self.assertListEqual(expected[1], result[1])
#         self.assertListEqual(expected[2], result[2])
#         self.assertListEqual(expected[3], result[3])
