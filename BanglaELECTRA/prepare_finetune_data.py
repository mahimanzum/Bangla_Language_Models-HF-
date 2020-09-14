import unicodedata
import re
import os
from sklearn.model_selection import train_test_split
from itertools import chain

def applyPatterns(line, patternMap):
    """
    Returns:
    patternFound(bool) : Whether any of the patterns were found in the line
    line(str) : Transformed line using pattern replacements
    """
    patternFound = False
    orignalLine = line
    for pattern, replacement in patternMap.items():
        line = line.replace(pattern, replacement)
        if orignalLine != line:
            patternFound = True

    return patternFound, line

def readReplacePatterns(filename):
    """
        Patterns should have the following form in each line:
        `Pattern:enReplacement(:optional bnReplacement)`

        Both Pattern and Replacement can contain arbitrary no of spaces.
        Be careful not to place unnecessary spaces.
        In absence of bnReplacement, bnReplacement = enReplacement
    """
    enPatternMap, bnPatternMap = {}, {}

    with open(filename, encoding='utf-8') as f:
        for line in f.readlines():
            try:
                splitLine = line.rstrip('\n').split(":")
                pattern = splitLine[0]
                enReplacement = splitLine[1]
                if len(splitLine) == 3:
                    bnPatternMap[pattern] = splitLine[2]
                else:
                    bnPatternMap[pattern] = enReplacement
                enPatternMap[pattern] = enReplacement
            except:
                continue
    
    return enPatternMap, bnPatternMap

EN_PATTERN_MAP, BN_PATTERN_MAP = readReplacePatterns('./replacePatterns.txt')

def normalize(text):
    def normalizePunct(text):
        REPLACE_UNICODE_PUNCTUATION = [
            (u"\u09F7", u"\u0964"),
            (u"，", u","),
            (u"、", u","),
            (u"”", u'"'),
            (u"“", u'"'),
            (u"∶", u":"),
            (u"：", u":"),
            (u"？", u"?"),
            (u"《", u'"'),
            (u"》", u'"'),
            (u"）", u")"),
            (u"！", u"!"),
            (u"（", u"("),
            (u"；", u";"),
            (u"」", u'"'),
            (u"「", u'"'),
            (u"０", u"0"),
            (u"１", u'1'),
            (u"２", u"2"),
            (u"３", u"3"),
            (u"４", u"4"),
            (u"５", u"5"),
            (u"６", u"6"),
            (u"７", u"7"),
            (u"８", u"8"),
            (u"９", u"9"),
            (u"～", u"~"),
            (u"’", u"'"),
            (u"…", u"..."),
            (u"━", u"-"),
            (u"〈", u"<"),
            (u"〉", u">"),
            (u"【", u"["),
            (u"】", u"]"),
            (u"％", u"%"),
        ]
        NORMALIZE_UNICODE = [
            ('\u00AD', ''),
            ('\u09AF\u09BC', '\u09DF'),
            ('\u09A2\u09BC', '\u09DD'),
            ('\u09A1\u09BC', '\u09DC'),
            ('\u09AC\u09BC', '\u09B0'),
            ('\u09C7\u09BE', '\u09CB'),
            ('\u09C7\u09D7', '\u09CC'),
            ('\u0985\u09BE', '\u0986'),
            ('\u09C7\u0981\u09D7', '\u09CC\u0981'),
            ('\u09C7\u0981\u09BE', '\u09CB\u0981'),
            ('\u09C7([^\u09D7])\u09D7', "\g<1>\u09CC"),
            ('\\xa0', ' '),
            ('\u200B', u''),  
            ('\u2060', u''),
            (u'„', r'"'),
            (u'“', r'"'),
            (u'”', r'"'),
            (u'–', r'-'),
            (u'—', r' - '),
            (r' +', r' '),
            (u'´', r"'"),
            (u'([a-zA-Z])‘([a-zA-Z])', r"\g<1>'\g<2>"),
            (u'([a-zA-Z])’([a-zA-Z])', r"\g<1>'\g<2>"),
            (u'‘', r"'"),
            (u'‚', r"'"),
            (u'’', r"'"),
            (u'´´', r'"'),
            (u'…', r'...'),
        ]
        FRENCH_QUOTES = [
            (u'\u00A0«\u00A0', r'"'),
            (u'«\u00A0', r'"'),
            (u'«', r'"'),
            (u'\u00A0»\u00A0', r'"'),
            (u'\u00A0»', r'"'),
            (u'»', r'"'),
        ]
        substitutions = [NORMALIZE_UNICODE, FRENCH_QUOTES, REPLACE_UNICODE_PUNCTUATION]
        substitutions = list(chain(*substitutions))

        for regexp, replacement in substitutions:
            text = re.sub(regexp, replacement, text, flags=re.UNICODE)
        
        text = re.sub(r'\s+', ' ', text)
    
        return text

    # text = check_output('perl remove-non-printing-char.perl | perl deescape-special-chars.perl', input=text, encoding='UTF-8', shell=True).strip()
    return normalizePunct(text)

def extended_normalize(line):
    _, line = applyPatterns(normalize(line.strip()).strip(), EN_PATTERN_MAP)
    _, line = applyPatterns(normalize(line.strip()).strip(), BN_PATTERN_MAP)
    
    return line

def writeFile(filename, lines):
    with open(filename, 'w') as f:
        for l in lines:
            l = [segment.replace('\t', ' ') for segment in l]
            print(*l, sep='\t', file=f)



def prepareSentimentData(inputDir):
    if '/BengFastText' in inputDir:
        originalList = []
        testList = []
        with open(f'{inputDir}/train/bangla.pos') as f:
            for line in f:
                originalList.append([extended_normalize(unicodedata.normalize('NFKC', extended_normalize(line.strip()))), 1])

        with open(f'{inputDir}/train/bangla.neg') as f:
            for line in f:
                originalList.append([extended_normalize(unicodedata.normalize('NFKC', extended_normalize(line.strip()))), 0])

        with open(f'{inputDir}/test/bangla.pos') as f:
            for line in f:
                testList.append([extended_normalize(unicodedata.normalize('NFKC', extended_normalize(line.strip()))), 1])

        with open(f'{inputDir}/test/bangla.neg') as f:
            for line in f:
                testList.append([extended_normalize(unicodedata.normalize('NFKC', extended_normalize(line.strip()))), 0])

        trainList, evalList = train_test_split(originalList, test_size=.1, random_state=3435)
        
        with open(f'{inputDir}/train.txt', 'w') as trainF, open(f'{inputDir}/eval.txt', 'w') as evalF, open(f'{inputDir}/test.txt', 'w') as testF:
            for line in trainList:
                print(line[0].replace('\t', ' '), line[1], sep='\t', file=trainF)

            for line in evalList:
                print(line[0].replace('\t', ' '), line[1], sep='\t', file=evalF)

            for line in testList:
                print(line[0].replace('\t', ' '), line[1], sep='\t', file=testF)

    elif '/SST' in inputDir:
        def normalizeFile(inputFile):
            lines = []
            with open(inputFile) as f:
                for line in f:
                    splitLine  = line.split('\t')
                    splitLine = [segment.strip() for segment in splitLine]
                    lines.append([extended_normalize(splitLine[0]), extended_normalize(splitLine[1])])
            return lines

        trainList = normalizeFile(f'{inputDir}/raw/train.txt')
        devList = normalizeFile(f'{inputDir}/raw/dev.txt')
        testList = normalizeFile(f'{inputDir}/raw/test.txt')

        writeFile(f'{inputDir}/train.txt', trainList)
        writeFile(f'{inputDir}/eval.txt', devList)
        writeFile(f'{inputDir}/test.txt', testList)




def prepareNLIData(inputDir):
    def normalizeFile(inputFile):
        lines = []
        with open(inputFile) as f:
            for line in f:
                splitLine  = line.split('\t')
                splitLine = [segment.strip() for segment in splitLine]
                # lines.append([unicodedata.normalize('NFKC', extended_normalize(splitLine[0])), unicodedata.normalize('NFKC', extended_normalize(splitLine[1])), splitLine[2]])
                lines.append([extended_normalize(splitLine[0]), extended_normalize(splitLine[1]), extended_normalize(splitLine[2])])
        return lines

    
    if '/mnli' in inputDir:       
        trainList = normalizeFile(f'{inputDir}/raw/train.bn')
        dev_matched_list = normalizeFile(f'{inputDir}/raw/dev_validation_matched.bn')
        dev_mismatched_list = normalizeFile(f'{inputDir}/raw/dev_validation_mismatched.bn')
        test_matched_list = normalizeFile(f'{inputDir}/raw/dev_test_matched.bn')
        test_mismatched_list = normalizeFile(f'{inputDir}/raw/dev_test_mismatched.bn')

        writeFile(f'{inputDir}/train.txt', trainList)
        writeFile(f'{inputDir}/eval_matched.txt', dev_matched_list)
        writeFile(f'{inputDir}/eval_mismatched.txt', dev_mismatched_list)
        writeFile(f'{inputDir}/test_matched.txt', test_matched_list)
        writeFile(f'{inputDir}/test_mismatched.txt', test_mismatched_list)





if __name__ == "__main__":
    prepareSentimentData('./inputs/finetune/sentiment/BengFastText')
    prepareSentimentData('./inputs/finetune/sentiment/SST')
    prepareNLIData('./inputs/finetune/nli/mnli')
             
             
             
