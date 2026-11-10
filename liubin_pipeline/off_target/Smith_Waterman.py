'''
based on algorithm from https://blog.csdn.net/qq_54501547/article/details/127981120
'''
import numpy as np
import pdb
import math

class LocalAligner_multi_findings_based_on_ss(object):
    def __init__(self, min_match, match, mismatch, gap):
        self.gapPen, self.mismatchPen, self.matchScore = gap, mismatch, match
        self.min_match_ntds = int(min_match)

    def calcTables(self):
        for i in range(1, len(self.q) + 1):
            for j in range(1, len(self.p) + 1):
                from_left_score = self.MatrixA[i][j - 1] + self.gapPen
                from_above_score = self.MatrixA[i - 1][j] + self.gapPen
                from_diag_score = self.MatrixA[i - 1][j - 1] + (self.matchScore if self.q[i - 1] == self.p[j - 1] else self.mismatchPen)
                left_up_diag_score = [from_left_score, from_above_score, from_diag_score, 0]
                self.MatrixA[i][j] = max(left_up_diag_score)
                self.MatrixB[i][j] = left_up_diag_score.index(self.MatrixA[i][j])
                # 0 == "left", 1 == "up" , 2 == "diagonal mismatch or match", 3 == 'none ending'

    def backtracking(self):
        path = []
        i, j = divmod(np.argmax(self.MatrixA), self.MatrixA.shape[1])
        m_stack = [(i, j)]
        a_stack = []
        while m_stack:
            row = m_stack[-1][0]
            col = m_stack[-1][1]
            if row <= 0 or col <= 0 or self.MatrixB[row][col] == 3:
                # ending, save searching traces
                path = m_stack.copy()
                return path
            else:
                if len(m_stack) > len(a_stack):
                    # check if the main stack and assistant stack have same length
                    a_stack.append([])
                    if self.MatrixB[row][col] == 2 and self.q[row - 1] == self.p[col - 1]:
                        a_stack[-1].append((row - 1, col - 1))
                    elif self.MatrixB[row][col] == 2:
                        a_stack[-1].append((row - 1, col - 1))
                    elif self.MatrixB[row][col] == 1:
                        a_stack[-1].append((row - 1, col))
                    elif self.MatrixB[row][col] == 0:
                        a_stack[-1].append((row, col - 1))
                # check if the assitant stack is empty, if not, visit its neighbors
                elif a_stack[-1] != []:
                    m_stack.append(a_stack[-1].pop())
                # if the assisatn stack is empty, then pop main and assitant smultaneously
                elif a_stack[-1] == []:
                    a_stack.pop()
                    m_stack.pop()

        return path

    def trace_to_base(self, path):
        seq_match1 = [self.q[path[0][0] - 1]]
        seq_match2 = [self.p[path[0][1] - 1]]
        connect = []
        connect.append("|" if seq_match1==seq_match2 else "*")
        for index in range(1, len(path)):
            if path[index][0] != path[index - 1][0] and path[index][1] != path[index - 1][1]:
                seq_match1.append(self.q[path[index][0] - 1])
                seq_match2.append(self.p[path[index][1] - 1])
                connect.append("|" if self.q[path[index][0] - 1] == self.p[path[index][1] - 1] else "*")
            elif path[index][0] == path[index - 1][0]:
                seq_match1.append("-")
                seq_match2.append(self.p[path[index][1] - 1])
                connect.append(" ")
            elif path[index][1] == path[index - 1][1]:
                seq_match1.append(self.q[path[index][0] - 1])
                seq_match2.append("-")
                connect.append(" ")
        seq_match1 = "".join(seq_match1)
        seq_match2 = "".join(seq_match2)
        connect = "".join(connect)
        return seq_match1, seq_match2, connect

    def reset_forwarding(self, seq_query=None, seq_subject=None):
        self.q, self.p = seq_query, seq_subject
        self.pmmt = self.min_match_ntds - math.ceil(abs(self.mismatchPen * (len(self.q) - self.min_match_ntds)) / self.matchScore)
        # possible min matched ntds
        self.required_min_score = self.matchScore * self.pmmt + self.mismatchPen * (len(self.q) - self.pmmt)

        # A table holds the scores. another table holds the direction in backtracking
        self.MatrixA = np.zeros(shape=[len(self.q) + 1, len(self.p) + 1])
        self.MatrixB = np.zeros(shape=[len(self.q) + 1, len(self.p) + 1])
        self.calcTables()
        result = []

        # loop for multiple outputs
        self.maxScore = self.MatrixA.max()
        while self.maxScore >= self.required_min_score:
            path = self.backtracking()
            path = path[::-1]
            self.finalQ, self.finalP, self.connect = self.trace_to_base(path)
            # clear the backtracking path values
            self.MatrixA[path[0][0]:path[-1][0]+self.pmmt-1, path[0][1]:path[-1][1]+self.pmmt-1] = 0
            if self.connect.count('|') >= self.min_match_ntds:
                result.append({'query': self.finalQ, 'subject': self.finalP, 'sub_index': path[0][1],
                                     'pairs': self.connect, 'num_pairs': self.connect.count('|')})
            self.maxScore = self.MatrixA.max()

        return result


def Homo_offtarget_Based_on_substr(query, subject, min_match=16):
    result = []
    for i in range(0, len(query)-min_match+1):
        sub_q = query[i:(i+min_match)]
        while True:
            pos = subject.find(sub_q)
            if pos==-1:
                break
            else:
                each_result = {'query': sub_q, 'subject': subject[pos:pos+min_match], 'sub_index': pos,
                               'pairs': '|'*min_match, 'num_pairs': min_match}
                result.append(each_result)
                subject = subject[:pos] + '.'*min_match + subject[pos+min_match:]

    return result


if __name__=='__main__':
    SmithWatermann = LocalAligner_multi_findings_based_on_ss(min_match=10, match=5, mismatch=-3, gap=-2)
    # seq_subject='CCGCCCCTTACGGCGCCGGAGAGATGGCGGAGTTGGACATCGGGCAGCACTGCCAGGTGGAGCATTGCCGGCAGCGAGATTTTCTTCCATTTGGAGTTGGACATCGTTGTGTGTGATGATTGTTCAGGAATATTTTGGTGACTGTAATCAATGAGAGACTGAAGACAGATCAACATACATCTTACCCATGCTCTTTCAAAGACTGTGCTGAGAGAGAACTTGTGGCAGTTATATGTCCTTATTGTGAGAAGAATTTTTGCCTGAGACACCGTCATCAGTCAGATCATGAGTGTGAAAAACTGGAAATCCCAAAGCCTCGAATGGCTGCCACTCAGAAACTTGTTAAAGACATTATTGATTCCAAGACAGGAGAAACAGCAAGTAAACGATGGAAAGGTGCCAAAAATAGTGAAACAGCTGCAAAGGTTGCATTGATGAAATTAAAGATGCATGCTGATGGCGATAAGTCATTACCACAGATGAGTAATCTGGAGTGAAGTATTTTTAAGAGCACTAACTTGACAGAAAGAATTTACTTTCAGGTTTTCTTACCTAAAGGGAGCAAAGAGAAGAGCAAACCAATGTTCTTTTGCCACCGATGGAGCATTGGAAAGGCCATAGACTTTGCCGCTTCTCTAGCCAGGCTTAAAAATGACAATAACAAATTTACAGCTAAGAAATTAAGGCTGTGTCACATTACTTCAGGAGAAGCCTTACCCTTGGATCATACTTTGGAAACCTGGATTGCTAAGGAGGATTGTCCTTTATATAATGGTGGAAATATAATCTTGGAATATCTTAATGATGAAGAACAATTCTGTAAAAATGTTGAATCTTACTTGGAATAGTCATTCAAAGATTCAAGTCAGAAATCACAGGGAAAATTCTATTACATGTTTAAGTCCATTTATTTTACTACAAATACTATTTTTGTAATTTTTAAAAGTTGCTTTTTAACTTATTTCCCCTTATGTCACATTTTAAATTATGAGTTTTCTATTAGAGTTTTTTGGTTTTAAGTTTAATTGTGTACGGATCCTCAATTATTAGCAAATTCTTATATTTAATTAGCTTTCATTAGTTTATGCTGTAGTGACAACAACAAAGATGTATTTCCTGGTGGGGGATGTCAGCTGTTGATTGGCTGTGACACTGCTTTGTGTCATCTTCTTACCATGATCAAAGGCAAAGGAAGGGATCTCTTTTGGGACATTGTGATTGTTTTAGCAGAGAGAGAAAGAGATGAAATACACTTCGGTTTTCTCTTAAAAGATGCATGTATCATACGTGCTTTAAGCAAGTCATGTGGCCAAGCCTAGCATCATGGAGCCAGAAAGTATAGCCTTGCTGTCTGTCTACATCATGATGTATAAATTGATATATCTACATGAATTATAGAAACTTAGAAGTGATCTTTATTCAGTCTTATAATTTTTACATGAAGAATCTTAGGCCTAGGAGGAGAAAATGATTTTCTTTCTATTACCTAACTAGATTGGGGCATATTTCTGATAAAGACCCACCTCTAGTGAGATTCATCTTTTTTGTTTGTGTGACTATATTCCATAGAGAAGAAAGATGGGATAGCTCAACTTCATTATATACCAAAGCAAAACACATGCCAAATGATGACTACATTTTACCAACATATTTAGACGAGTATTCTTGACTAGTGTTTACTATCTATACCCCCAAAACTACTACTATATAGACAGAATGGAAAGTATTTCTATTTGTCCTTTTTTTGTTTTCTGTTCTAATTGTCAGGGACATATGTAGTGGCTATAGGTTTACTTAAAAGGAATAAATTTGGAATGCTCATTTGGTTTATTTCCTTTCTAATTCTTCATATTATTTTTATGTGAGGAATGGAGATAATTTCCAACTTTTATTATGTTCTTGTACTTTCATTCATTAGCATAGTTATTTGAGTTCCCACTAAATGTAGGGATTTGTCCTAATTGCCATGATTAATACAATGATGATATGTATACATACAGGTATACCTTGGAAATATTGTGGGTTCAGTTTCAGACTACCACACAAAGCAAATATTGGAGTATATTGAGCCAGACAAATTTTTTTTTGTTTCCCAGGGCATATAAAAATTATGTTTACACTATACTATAGTCTATTTAATGTGCAATAGCATTATGTCTAAAAAATGTATATACTTTAAGTAAAAAAATACTTTATTGCTAAAAAATGCTAAAAAAAAAAAAAAAAAA'
    # print('test by improved smith-waternman:\n')
    # result = SmithWatermann.reset_forwarding(seq_query='TGAAGGTAATCCGGAGTGA', seq_subject=seq_subject)
    # for ele in result:
    #     print(ele)
    # print('\ntest by substr:\n')
    # result = Homo_offtarget_Based_on_substr(query='TGAAGGTAATCCGGAGTGA', subject=seq_subject, min_match=16)
    # for ele in result:
    #     print(ele)

    # seq_subject="CCGCCCCTTACGGCGCCGGAGAGATGGCGGAGTTGGACATCGGGCAGCACTGCCAGGTGGAGCATTGCCGGCAGCGAGATTTTCTTCCATTTGGAGTTGGACATCGTTGTGTGTGATGATTGTTCAGGAATATTTTGGTGACTGTAATCAATGAGAGACTGAAGACAGATCAACATACATCTTACCCATGCTCTTTCAAAGACTGTGCTGAGAGAGAACTTGTGGCAGTTATATGTCCTTATTGTGAGAAGAATTTTTGCCTGAGACACCGTCATCAGTCAGATCATGAGTGTGAAAAACTGGAAATCCCAAAGCCTCGAATGGCTGCCACTCAGAAACTTGTTAAAGACATTATTGATTCCAAGACAGGAGAAACAGCAAGTAAACGATGGAAAGGTGCCAAAAATAGTGAAACAGCTGCAAAGGTTGCATTGATGAAATTAAAGATGCATGCTGATGGCGATAAGTCATTACCACAGATGAGTAATCTGGAGTGAAGTATTTTTAAGAGCACTAACTTGACAGAAAGAATTTACTTTCAGGTTTTCTTACCTAAAGGGAGCAAAGAGAAGAGCAAACCAATGTTCTTTTGCCACCGATGGAGCATTGGAAAGGCCATAGACTTTGCCGCTTCTCTAGCCAGGCTTAAAAATGACAATAACAAATTTACAGCTAAGAAATTAAGGCTGTGTCACATTACTTCAGGAGAAGCCTTACCCTTGGATCATACTTTGGAAACCTGGATTGCTAAGGAGGATTGTCCTTTATATAATGGTGGAAATATAATCTTGGAATATCTTAATGATGAAGAACAATTCTGTAAAAATGTTGAATCTTACTTGGAATAGTCATTCAAAGATTCAAGTCAGAAATCACAGGGAAAATTCTATTACATGTTTAAGTCCATTTATTTTACTACAAATACTATTTTTGTAATTTTTAAAAGTTGCTTTTTAACTTATTTCCCCTTATGTCACATTTTAAATTATGAGTTTTCTATTAGAGTTTTTTGGTTTTAAGTTTAATTGTGTACGGATCCTCAATTATTAGCAAATTCTTATATTTAATTAGCTTTCATTAGTTTATGCTGTAGTGACAACAACAAAGATGTATTTCCTGGTGGGGGATGTCAGCTGTTGATTGGCTGTGACACTGCTTTGTGTCATCTTCTTACCATGATCAAAGGCAAAGGAAGGGATCTCTTTTGGGACATTGTGATTGTTTTAGCAGAGAGAGAAAGAGATGAAATACACTTCGGTTTTCTCTTAAAAGATGCATGTATCATACGTGCTTTAAGCAAGTCATGTGGCCAAGCCTAGCATCATGGAGCCAGAAAGTATAGCCTTGCTGTCTGTCTACATCATGATGTATAAATTGATATATCTACATGAATTATAGAAACTTAGAAGTGATCTTTATTCAGTCTTATAATTTTTACATGAAGAATCTTAGGCCTAGGAGGAGAAAATGATTTTCTTTCTATTACCTAACTAGATTGGGGCATATTTCTGATAAAGACCCACCTCTAGTGAGATTCATCTTTTTTGTTTGTGTGACTATATTCCATAGAGAAGAAAGATGGGATAGCTCAACTTCATTATATACCAAAGCAAAACACATGCCAAATGATGACTACATTTTACCAACATATTTAGACGAGTATTCTTGACTAGTGTTTACTATCTATACCCCCAAAACTACTACTATATAGACAGAATGGAAAGTATTTCTATTTGTCCTTTTTTTGTTTTCTGTTCTAATTGTCAGGGACATATGTAGTGGCTATAGGTTTACTTAAAAGGAATAAATTTGGAATGCTCATTTGGTTTATTTCCTTTCTAATTCTTCATATTATTTTTATGTGAGGAATGGAGATAATTTCCAACTTTTATTATGTTCTTGTACTTTCATTCATTAGCATAGTTATTTGAGTTCCCACTAAATGTAGGGATTTGTCCTAATTGCCATGATTAATACAATGATGATATGTATACATACAGGTATACCTTGGAAATATTGTGGGTTCAGTTTCAGACTACCACACAAAGCAAATATTGGAGTATATTGAGCCAGACAAATTTTTTTTTGTTTCCCAGGGCATATAAAAATTATGTTTACACTATACTATAGTCTATTTAATGTGCAATAGCATTATGTCTAAAAAATGTATATACTTTAAGTAAAAAAATACTTTATTGCTAAAAAATGCTAAAAAAAAAAAAAAAAAA"
    # result = SmithWatermann.reset_forwarding(seq_query="TGAAGGTAATCCGGAGTGA", seq_subject=seq_subject)
    seq_subject = "TCGTACTCTAACGAT"
    result = SmithWatermann.reset_forwarding(seq_query="AACGTACTCAAGTCT", seq_subject=seq_subject)
    for ele in result:
        print(ele)

    pdb.set_trace()