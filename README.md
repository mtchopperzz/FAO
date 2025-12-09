このプログラムはAlex Vinogradov (https://scholar.google.com/citations?user=fVm6Lv4AAAAJ&hl=en; https://github.com/avngrdv) に作られたFastqProcessorより発展されたものです。
彼の元生徒として、私は大学院課程での彼の援助とアドバイス、そして FastProcessor をオープンソース化してくれたことに深く感謝し、尊敬しています。


# FastqProcessor-Antibody optimized (FAO)
抗体取得のワークフローにおいて、NGSから取得したfastqファイルを処理し、抗体配列を抽出する

FAOは、FastqProcessorに基づいて、抗体取得のために改造されました。
Fastqからアミノ酸は配列に翻訳し、Fab・可変領域などの配列を抽出・統計して結果ファイルを産生する。
コアの機能としては、事前にプログラムに入れ込んだライブラリーのデザインにしたがって、ほしい配列を探す・抽出することです。
それに加えて、NGSにあり得るノイズリード・低品質リードを除去したり、収束率を計算したり、抽出した配列の長さを統計したりする機能も備えています。


# Motivation
オリジナルバージョンのFastqAnalyserは、mRNA displayのために作成されたものとなり、有数な合成テンプレートの混合物からできたライブラリーがスクリニーングを経て、DNA配列やアミノ酸配列の変化を分析するに使用されます。mRNAライブラリーが事前にきちんと設計されたものであるため、如何にライブラリーに現れるべきではない配列を除去するかに工夫をしています。

# 依頼環境
テスト済み:\
python 3.8.5\
numpy 1.19.5\
pandas 1.2.4\
matplotlib 3.3.2\
\

#　ログ
## Dec, 09, 2025
Uploaded.

# 使用方法

## run in terminal
python ONO_1234_analysis.py -c /PATH/TO/CONFIG/ONO_1234_config.py

## analysis.py
データの解析を初期化し、操作をqueueに入れてプロセスを実行するための.pyスクリプト(e.g., ONO_1234_analysis.py)を作成する必要があります。
config, ParserとPipeline（pipelineはparserをqueueに入れて実行するために使用されます）をDispatcherとHandlerを入れ込みます：

```python
    #import prerequisities
    import argparse, importlib, importlib.util, sys, os
    from utils.ProcessHandlers import Pipeline, FastqParser
    from utils.Dispatcher import Dispatcher
    
    #config file holds the information about library designs and
    #other parser instructions (where to look for data, where to save results etc)
    def load_config(cfg_arg: str):
      if os.path.isfile(cfg_arg):
          spec = importlib.util.spec_from_file_location("config", cfg_arg)
          module = importlib.util.module_from_spec(spec)
          sys.modules["config"] = module      # <-- critical line
          spec.loader.exec_module(module)
      else:
          module = importlib.import_module(cfg_arg)
          sys.modules["config"] = module
      return module
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs.ab_lib_A",
                    help="module or path of config to load "
                         "(default: configs.ab_lib_A)")
    args = ap.parse_args()

    config = load_config(args.config)

    dispatcher = Dispatcher(config)
    pip, par  = dispatcher.dispatch_handlers((Pipeline, FastqParser))

    
    #initialize a dispatcher object; dispatcher is strictly speaking
    #not necessary, but it simplifies initialization of data handlers
    dispatcher = Dispatcher(config)    
    
    #a list of handlers to initialize; pipeline should always be included
    #if NGS data parsing is the goal, FastqParser will do most of the work
    handlers = (Pipeline, FastqParser)
    
    #initialize the handlers
    pip, par = dispatcher.dispatch_handlers(handlers)
```

FastqParserは、入出力およびデータ操作のためのすべてのメソッドを保持しています。これらのfunctionはpipelineにqueueに入れることができます。
例えば、データの取得とDNAからpeptideへの翻訳操作をqueueに入れるには：

```python
pip.enque([par.fetch_gz_from_dir(), par.translate()])
```

これにより、piplineに2つの操作が追加されますが、データの取得や翻訳はまだ行われません。追加のメタParameterを含むいくつかの操作については、この時点でParameterの検証も行われます。最後に、

```python
pip.run(save_summary=True)
```

を実行すると、queueに入れられた操作が指定された順序で1つずつ実行され、データが次の操作に渡されます。これで完了です。期待される出力がある場合は、config.pyで指定されたディレクトリに書き込まれます。必要に応じてログファイルもオプションで書き込まれることがあります。


## config
.fastqまたは.fastq.gzファイルのセットを解析するには、まずconfigファイル（config.py）を編集または作成する必要があります。
config.pyには、DNAからタンパク質への翻訳テーブル（カスタマイズ可能）、ライブラリ設計情報（peptideおよびDNAレベルの両方）、入力および出力フォルダの場所、ログParameterなどのその他の情報を含める必要があります。

## LibraryDesign
LibraryDesignは、Parserが期待するLibraryの種類を指定する重要なオブジェクトです。このオブジェクトは、統一されたロジックを使用して任意のDNAおよびpeptideライブラリに関する情報を保持できます。ランダム化されたアミノ酸/塩基（以下、トークン）は数字（0-9）で示され、ランダム化の対象ではないトークン（リンカー配列など）は標準の1文字エンコーディング（DNAの場合はA、C、T、G、peptideの場合はA、C、Dなど）で示されます。連続したランダムまたは固定トークンのストレッチは、テンプレートシーケンス内の「領域」を構成します。例えば：

```
                seq:      ACDEF11133211AWVFRTQ12345YTPPK
             region:      [-0-][---1--][--2--][-3-][-4-]
        is_variable:      False  True   False True False
```

領域の割り当ては自動的に行われます。上記の例では、ライブラリには5つの領域が含まれており、3つは「定数領域」であり、2つは「可変領域」です。可変領域トークンに使用される数字は定義され、特定のトークンセットに対応する1つの数字が割り当てられます。例えば、NNKコドンはすべての20種類のアミノ酸をコードしますが、NNCコドンは15種類のみです。したがって、NNKコドンから派生したすべてのアミノ酸は1つの数字でエンコードされ、NNCでエンコードされた位置には別の数字が割り当てられます。LibraryDesignは、可変サイズの可変領域を持つライブラリを考慮するために、異なる長さの複数のテンプレートを取ることができます。
以下はLibraryDesignの初期化の例です：

```python
lib = LibraryDesign(
    
                templates=[
                            'ACDEF11133211AWVFRTQ12345YTPPK',
                            'ACDEF11122211AWVFRTQ12345YTPPK',
                            'ACDEF11111211AWVFRTQ12345YTPPK',
                          ],
        
                monomers={
                          1: ('A', 'C', 'D', 'E', 'F', 'G', 'H'),
                          2: ('M'),
                          3: ('C')
                         },
                
                lib_type='pep'
                    
                   )
```

可変位置が単一のアミノ酸をエンコードできることに注意してください（アミノ酸2と3）。この方法で、特定のライブラリを表現する際にかなりの柔軟性があります。LibraryDesignオブジェクトを初期化する際には、いくつかのルールに従う必要があります：

1.	渡されたすべてのテンプレートのトポロジーは同一でなければなりません。トポロジーとは、領域の総数と可変領域の総数です。基本的に、テンプレートは可変領域の内部構成のみが異なるべきです。
2.	すべての可変領域モノマーは翻訳テーブルでエンコードされている必要があります（またはDNAライブラリの場合は4つの標準DNA塩基のいずれかである必要があります。塩基N、Kなどは数字に変換されるべきです）。
3.	Parserには2つのLibraryDesignオブジェクトを作成する必要があります（lib_type='dna'とlib_type='pep'）。
4.	Templateの長さは実際の配列長さに一致する必要があります。fuzzyシリーズのparだけはvrの長さを無視します。その場合は、可変領域を適当な111にしてすればよいです。
5.	DNAテンプレートを使用してfilterやfetchを操作する場合(where=:'dna')だけDNAの設計をきちんとconfigに登録する必要があります。アミノ酸配列だけ操作する場合はDNAテンプレートをデフォルトのままにしてもよいです。
