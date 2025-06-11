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
それに対して、抗体取得にNGSを駆使する場合では、できるだけ多くのバインダーを抽出しなければなりませんので、翻訳の開始コドン、DNAテンプレートのパターン、可変領域の長さなどを無視できるtranslate_all_frame、fuzzy_filterを追加しました。

# 依頼環境
テスト済み:\
python 3.8.5\
numpy 1.19.5\
pandas 1.2.4\
matplotlib 3.3.2\
\

#　ログ
## Jun, 05, 2025
fastqから抗体のアミノ酸配列を抽出するワークフローを追加
trnaslate_all_framesですべての翻訳パターンを生成
シグナルpeptideやCHを除去するためにfuzzyシリーズを追加
## Important!!!
Jun, 06, 2025の時点で、DNA配列を操作する機能の動作確認は未完成です。
そのため、where='DNA'の使用はお控えください。

# To-do List
DNA操作の動作確認
IMGTから動物種のGermlineをconfigに取り込む
Numberingはどう実現するか

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

## データ（趙の追記：numpyの使い方は分からないため、この部分の改造はまた有識者にお願いいたします。）
解析中、データはDataオブジェクトのインスタンスとして保存されます。Dataは個々のサンプルのコンテナであり、SequencingSampleオブジェクトとして保存されます。原則として、任意の数のDNAシーケンスがサンプルになることができますが、実際にはほとんどの場合、1つのサンプル＝1つの.fastqファイルです。SequencingSampleオブジェクトには4つの公開属性があります：

- SequencingSample.name: サンプル名（strとして）
- SequencingSample.D: DNAシーケンスのリスト（Noneとして設定可能）
- SequencingSample.Q: Qスコアシーケンスのリスト（Noneとして設定可能）
- SequencingSample.P: peptideシーケンスのリスト（Noneとして設定可能）

これらのリストはnumpy配列として保存されます：FastqParser.transform()またはFastqParser.translate()を呼び出す前は1D配列であり、その後は常に2D配列です。形状：（エントリー数、シーケンス長）。異なるリードのシーケンスが異なる長さを持つ可能性があるため、配列は最も長いシーケンスにパディングされます。

プロセス全体を通じて各配列のエントリー数は同じに保たれますが、1つ以上の属性がNoneに設定されている場合を除きます。特定のフィルタリングルーチン（例えば、FastqParser.q_score_filt()）は単一の配列（この例ではSequencingSample.Q）に作用しますが、結果として3つの配列すべてのエントリーが破棄/保持されます。

LibraryDesignで指定されたテンプレートの数と種類によって、SequencingSampleの任意のエントリーは原則として複数のテンプレートと同時に互換性がある可能性があります。どのエントリーがどの種類のテンプレートに割り当てられるべきかを判断することがParserの主な目的の1つです。最初は（つまり、FastqParser.translate()を呼び出した直後）、Parserはすべてのシーケンスが指定されたすべてのテンプレートと互換性があると見なします。フィルタリングが進むにつれて、操作ごとにこの互換性が洗練されます。特定のSequencingSampleの割り当ての状態を「サンプルの内部状態」と呼びます。いくつかの操作（例えば、FastqParser.fetch_at()）は、どのテンプレートがどのエントリーに関連付けられるべきかを正確に知る必要があります。複数の可能なテンプレートと互換性のあるエントリーを見つけた場合、サンプルの内部状態を「崩壊」させ、1つの互換性のあるテンプレートを選択し、他のすべてを非互換として割り当てます。サンプルの内部状態を崩壊させることができる操作の詳細については、以下の操作リストを参照してください。一般的に、これらはフィルタリング操作の後に呼び出されるべきです。

## 操作

### FastqParser.fetch_fastq_from_dir()
stream_from_gz()はgzファイルを一個読み込んで、処理して、完了してから次のファイルを取り込むので、メモリーが足りないときに使います。
    
        sequencing_dataディレクトリ（config.pyで指定された）からすべての.fastqファイルを取得します。
        ワークフローの最初の操作として呼び出されるべきです。
        
            Parameter:
                    なし
        
            Return:
                    Dataのインスタンスとして取得されたFastqデータ

### FastqParser.fetch_gz_from_dir()
    
        sequencing_dataディレクトリ（config.pyで指定された）からすべての.fastq.gzファイルを取得します。
        ワークフローの最初の操作として呼び出されるべきです。
        
            Parameter:
                    なし
        
            Return:
                    Dataのインスタンスとして取得されたFastqデータ

### FastqParser.revcom()
    
        Data内の各サンプルについて、DNAシーケンスの逆相補を取得し、対応するQスコアのシーケンスを逆にします。
        使用する場合は、取得操作の直後にqueueに入れ、翻訳などの下流の操作の前に行うべきです。
        
        Parameter:
                なし
    
        Return:
                逆相補されたDNAと逆転されたQスコア情報を保持する変換されたDataオブジェクト

### FastqParser.transform()
    
        FastqParser.translate()の使用を推奨します。使用する場合は、データを取得した後、（オプションで）FastqParser.revcom()操作を実行した後に呼び出すべきです。
        データを下流の操作に適した表現に変換します。
        
        Parameter:
                なし
    
        Return:
                変換されたDataオブジェクト


### FastqParser.translate(force_at_frame=None)
    	Data内の各サンプルについて、DNAシーケンスデータのインシリコ翻訳を行います。
    	この操作は、1リードあたり1つのORFを持つNGSデータを対象としていますが、長い複数のORFを持つサンプルには適していません。
             
        この操作は、データを取得した後、（オプションで）FastqParser.revcom()を実行し、フィルタリングルーチンの前に呼び出すべきです。
        
        翻訳を実行するだけでなく、この操作はデータを下流の操作に適した表現に変換します。
        
        Parameter:
                force_at_frame: Noneの場合、通常のORF検索が行われます。通常のORF検索は、ATGコドンの上流にShine-Dalgarno配列を探すことを含みます（ORFを示す正確な5'-UTR配列はconfig.pyで指定されています）。
                                												
                                Noneでない場合、0、1、2の値を取ることができます。これにより、SD配列の有無にかかわらず、指定されたフレームで翻訳が強制的に開始されます。
                                
                                例えば：        
                                DNA: TACGACTCACTATAGGGTTAACTTTAAGAAGGA
                   force_at_frame=0  ----------> 
                    force_at_frame=1  ---------->
                     force_at_frame=2  ---------->
                                 
              stop_readthrough:	bool（True/False; デフォルト: False）。Trueの場合、ストップコドンに遭遇した後も対応するリードの3'-末端まで翻訳が続行されます。最後に遭遇したコドンが1または2塩基欠けている場合、peptideシーケンスのC末端に"_"アミノ酸が追加されます。
                                
                                Falseの場合、操作は真のORFシーケンスを返します。この場合、ストップコドンが欠けているORFからのpeptideシーケンスは、C末端に"+"アミノ酸が付けられます。
                                
                                リード内にストップコドンがないORFにはTrueをフラグするべきです。
				 
        Return:
                peptideシーケンス情報を含むDataオブジェクト


### FastqParser.translate_all_frame(force_at_frame=None)
    	Data内の各サンプルについて、DNAシーケンスデータの全部の三つのframeを翻訳する。

        同じDNA配列から三つのpeptide配列が派生するため、Dataフォーマットを保つために、同じDNAとQscoreを翻訳された三つのpeptideに付与する。
             
        この操作は、データを取得した後、（オプションで）FastqParser.revcom()を実行し、フィルタリングルーチンの前に呼び出すべきです。
        
        翻訳を実行するだけでなく、この操作はデータを下流の操作に適した表現に変換します。
        
        Parameter:
              stop_readthrough:	bool（True/False; デフォルト: False）。Trueの場合、ストップコドンに遭遇した後も対応するリードの3'-末端まで翻訳が続行されます。最後に遭遇したコドンが1または2塩基欠けている場合、peptideシーケンスのC末端に"_"アミノ酸が追加されます。
                                
                                Falseの場合、操作は真のORFシーケンスを返します。この場合、ストップコドンが欠けているORFからのpeptideシーケンスは、C末端に"+"アミノ酸が付けられます。
                                
                                リード内にストップコドンがないORFにはTrueをフラグするべきです。
				 
                                抗体取得の場合では、シーケンス範囲内に終止コドンがない可能性はあるため、デフォルトでFalseにします。
        Return:
                peptideシーケンス情報を含むDataオブジェクト



### FastqParser.len_filter(where=None, len_range=None)
    
        Data内の各サンプルについて、指定されたライブラリ設計よりも長い/短いシーケンスをフィルタリングします。
        または、エントリー（NGSリード）をこの範囲外にフィルタリングするために、シーケンスの長さ範囲をオプションで指定することができます。
        
        Parameter:
                   where: 'dna'または'pep'を指定して、操作がどのデータセットで動作するかを指定します。
						  
               len_range: Noneの場合、フィルタリングはライブラリ設計ルールに従って行われます。
                          または、取得する長さ範囲を指定する2つの整数のリスト。
					 
        Return:
                長さフィルタリングされたデータを含む変換されたDataオブジェクト
				
### FastqParser.cr_filter(where=None, loc=None, tol=1)
    
        Data内の各サンプルについて、完全な定数領域を含まないシーケンスをフィルタリングします。
        ライブラリ設計仕様外のアミノ酸を含む定数領域を持つエントリー（NGSリード）は破棄されます。    
	
        Parameter:
                   where: 'dna'または'pep'を指定して、操作がどのデータセットで動作するかを指定します。
						  
                     loc: 処理する定数領域を指定する整数のリスト。

                     tol: int; 定数領域が破棄される前に許容される最大変異数を指定します。
                          上記のライブラリの場合
                          
                seq:      ACDEF11133211AWVFRTQ12345YTPPK
             region:      [-0-][---1--][--2--][-3-][-4-]
        is_variable:      False  True   False True False
                          
                          cr_filter(where='pep', loc=[2], tol=1)を呼び出すと、
                          'AWVFRTQ'領域に1つ以上の変異を含むすべてのシーケンスが破棄されます。
                          定数領域の挿入/削除はParserによって検証されません。					  
					 
        Return:
                完全な定数領域を持つエントリーを含む変換されたDataオブジェクト
 				
### FastqParser.vr_filter(where=None, loc=None, sets=None)
    
        Data内の各サンプルについて、完全な可変領域を含まないシーケンスをフィルタリングします。
        ライブラリ設計仕様外のアミノ酸を含む可変領域を持つエントリー（NGSリード）は破棄されます。
    
        Parameter:
                   where: 'dna'または'pep'を指定して、操作がどのデータセットで動作するかを指定します。
						  
                     loc: 処理する可変領域を指定する整数のリスト。

                    sets: チェックするモノマーサブセットのリスト。
                          上記のライブラリの場合
                          
                seq:      ACDEF11133211AWVFRTQ12345YTPPK
             region:      [-0-][---1--][--2--][-3-][-4-]
        is_variable:      False  True   False True False
                          
                          5つの異なる可変アミノ酸があります：
                          1, 2, 3, 4, 5。configファイルはこれらの数字に対して許可される特定のアミノ酸を指定します。
                          <vr_filter>操作は、各可変位置が「許可された」モ


### FastqFraser.cr_filter_fuzzy(where=None,loc=None,tol=0)

        Data内の各サンプルについて、完全な定数領域を含まないシーケンスをフィルタリングします。
        ライブラリ設計仕様外のアミノ酸を含む定数領域を持つエントリー（NGSリード）は破棄されます。 
        cr_filterより、可変領域の長さを無視しているため、IMGTのテンプレートや、抗体の定常領域を指定して、抗体らしいpeptide配列を持つNGSリード以外の行を除去します(translate_all_frameで派生したものなど)。   
	
        Parameter:
                   where: 'dna'または'pep'を指定して、操作がどのデータセットで動作するかを指定します。
						  
                     loc: 処理する定数領域を指定する整数のリスト。

                     tol: int; 定数領域が破棄される前に許容される最大変異数を指定します。
                          上記のライブラリの場合
                          
                seq:      1111ACDEF11133211AWVFRTQ12345YTPPK1111
             region:      [0-][-1-][---2--][--3--][-4-][-5-][6-]
        is_variable:      TrueFalse  True   False True FalseTrue
                          
                          cr_filter_fuzzy(where='pep', loc=[1,3,5], tol=1)を呼び出すと、
                          'ACDEF','AWVFRTQ','YTPPK'領域を全部に持たず、1つ以上の変異を含むすべてのシーケンスが破棄されます。
                          定数領域の挿入/削除はParserによって検証されません。					  
					 
        Return:
                完全なvrを持つエントリーを含む変換されたDataオブジェクト


### FastqFraser.mask_regions_fuzzy(where=None,loc=None,mode='cr',tol=0,mask_token='*')

        Data内の各サンプルについて、指定された領域のpeptide配列を指定したTokenに変換し、maskします。  
	
        Parameter:
                   where: 'dna'または'pep'を指定して、操作がどのデータセットで動作するかを指定します。
						  
                     loc: 処理する定数領域を指定する整数のリスト。

                     mode: maskするregionを'vr'または'cr'に指定します。（vrとcrのmaskを別々のcodeで実現しているため、modeを指定する必要があります。違うregionをmaskする場合は2回callしてください。）

                     tol: int; 定数領域が破棄される前に許容される最大変異数を指定します。

                     mask_token: モノマーを置き換える文字を指定します。デフォルトでは'*'です。
                          
                seq:      1111ACDEF11133211AWVFRTQ12345YTPPK1111
             region:      [0-][-1-][---2--][--3--][-4-][-5-][6-]
        is_variable:      TrueFalse  True   False True FalseTrue
                          
                          mask_regions_fuzzy(where='pep', loc=[0,6], mode='vr',mask_token='*')を呼び出すと、
                          loc 1とloc 6のvrがすべて'*'に変換されます。					  
					 
        Return:
                指定されたregionがmaskされたDataオブジェクト

### FastqParser.filt_ambiguous(where=None)

        各Dataサンプルについて、曖昧なトークンを含まないシーケンスをフィルタリングします。DNAの場合、これらはIllumina NGSルーチンがベースコール中に時折割り当てる「N」ヌクレオチドです。ペプチドの場合、これらは翻訳テーブル仕様外のアミノ酸を含む任意のシーケンスです。

        Parameter:
                where: 操作がどのデータセットで動作するかを指定するための'dna'または'pep'
        
        Return:
                曖昧なトークンを含まないエントリーを持つ変換されたDataオブジェクト

### FastqParser.drop_data(where=None)
        各Dataサンプルについて、'where'で指定されたデータセットを削除します。Dataオブジェクトのドキュメントを参照してください。

        Parameter:
                where: 削除するデータセットを指定するための'dna'、'pep'、または'q'

        Return:
                削除されたデータセットを持たない変換されたDataオブジェクト

### FastqParser.q_score_filt(minQ=None, loc=None)

        各Dataサンプルについて、指定された閾値minQ以下のQスコアに関連付けられたシーケンスをフィルタリングします。

        Parameter:
                loc: 操作が処理する領域を指定する整数のリスト
                minQ: locで指定された領域内のすべてのQスコアがこの値以上である必要があります。それ以外は破棄されます
        
        Return:
                変換されたDataオブジェクト

### FastqParser.fetch_at(where=None, loc=None)

        各Dataサンプルについて、'where'で指定されたデータセットの中から'loc'で指定された領域を取得し、他のシーケンス領域を破棄します。
        サンプルの内部状態を崩壊させます。Dataオブジェクトのドキュメントを参照してください。

        Parameter:
                where: 操作がどのデータセットで動作するかを指定するための'dna'または'pep'
                loc: 取得する領域を指定する整数のリスト
        
        Return:
                変換されたDataオブジェクト

### FastqParser.fetch_at_fuzzy(where=None, loc=None, tol=0, pad="", keep_design="True")
        Fuzzy versionのfetch_at
        異なるリードで可変領域（VR）が長く/短くても動作します。
        定数ブロックはテンプレートと最大tolの置換で異なる場合があります。
        locに指定された領域のみを保持し、その他はすべて破棄します（whereにしてされなかったdataには影響を与えません）。

        Parameter
                where : {"pep", "dna"}, 操作を行うデータセット。
                loc : list[int], LibraryDesign.locから保持する領域インデックス（0ベース）。
                tol : int, 定数領域を識別する際に許容される最大のブロックごとのハミング距離。
                pad : str, 結果が依然として矩形のndarrayであるように右側にパディングするために使用される文字。
                keep_design : bool, Trueの場合、LibraryDesignオブジェクトは切り詰められません。そうでない場合、保持された領域に縮小され、下流のloc値が有効なままになります。

        Return:
                変換されたDataオブジェクト

### FastqParser.unpad()
        各Dataサンプルについて、D、Q、P配列のパディングを解除します。各配列について、すべての値がパディングトークンである列を削除します。Dataオブジェクトのドキュメントを参照してください。

        Parameter:
                なし
        Return:
        変換されたDataオブジェクト

### FastqParser.len_summary(where=None, save_txt=False)

        各Dataサンプルについて、ペプチド/DNAシーケンス長の分布を計算し（'where'で指定）、config.pyで指定されたParser出力フォルダに結果のヒストグラムをプロットします。オプションで、データをtxtファイルに書き込むこともできます。

        Parameter:
                where: 操作がどのデータセットで動作するかを指定するための'dna'または'pep'

                save_txt: Trueの場合、データは.pngおよび.svgプロットと同じフォルダに保存されたtxtファイルに書き込まれます
        Return:
                Dataオブジェクト（変換なし）

### FastqParser.convergence_summary(where=None)
        
        各Dataサンプルについて、シーケンスレベルで基本的なライブラリ収束解析を行います。正規化されたシャノンエントロピーと位置ごとのシーケンス保存を計算します。結果をconfig.pyで指定されたParser出力フォルダにプロットします。

        Parameter:
        where: 操作がどのデータセットで動作するかを指定するための'dna'または'pep'
        Return:
        Dataオブジェクト（変換なし）

### FastqParser.freq_summary(where=None, loc=None, save_txt=False)
        トークンレベルで基本的なライブラリ収束解析を行います。各Dataサンプルについて、データセット内の各トークンの頻度を計算します。結果をconfig.pyで指定されたParser出力フォルダにプロットします。オプションで、データをtxtファイルに書き込むこともできます。

        Parameter:
                where: 操作がどのデータセットで動作するかを指定するための'dna'または'pep'

                loc: 分析する領域を指定する整数のリスト。この場合、操作はサンプルの内部状態を崩壊させます（Dataオブジェクトの説明を参照）
                
                'all': 全体のシーケンスに対して同じ統計を取得します。この場合、操作はサンプルの内部状態を崩壊させません
                
                save_txt: Trueの場合、データは.pngおよび.svgプロットと同じフォルダに保存されたtxtファイルに書き込まれます
        Return:
                Dataオブジェクト（変換なし）

### FastqParser.q_summary(loc=None, save_txt=False)
        各Dataサンプルについて、基本的なQスコア統計を計算します。'loc'で指定された領域の各位置について、Qスコアの平均と標準偏差を計算します。結果をconfig.pyで指定されたParser出力フォルダにプロットします。オプションで、データをtxtファイルに書き込むこともできます。

        Parameter:
                loc: 分析する領域を指定する整数のリスト。この場合、操作はサンプルの内部状態を崩壊させます（Dataオブジェクトの説明を参照）
                'all': 全体のQスコア配列に対して同じ統計を取得します。この場合、操作はサンプルの内部状態を崩壊させません
                save_txt: Trueの場合、データは.pngおよび.svgプロットと同じフォルダに保存されたtxtファイルに書き込まれます
        Return:
                Dataオブジェクト（変換なし）

### FastqParser.count_summary(where=None, top_n=None, fmt=None)
        各Dataサンプルについて、'where'で指定されたデータセット内で各ユニークシーケンスが見つかる回数をカウントします。結果はconfig.pyで指定されたParser出力フォルダにファイルとして書き込まれます。

        Parameter:
                where: 操作がどのデータセットで動作するかを指定するための'dna'または'pep'
                top_n: Noneの場合、完全なサマリーが作成されます。整数が渡された場合、上位top_nシーケンス（カウント順）のみがファイルに書き込まれます
                fmt: 出力ファイルの形式。サポートされている値は'csv'と'fasta'
        Return:
                Dataオブジェクト（変換なし）

### FastqParser.template_summary(where=None)
        各Dataサンプルについて、'where'で指定されたデータセットと対応するライブラリテンプレートとの一致数を計算します。結果はconfig.pyで指定されたParser出力フォルダにファイルとして書き込まれます。
        言い換えれば、データセットシーケンスがどのライブラリから来ているかを要約します。この操作は"_internal_state_summary"とも呼ばれることがあります。

        Parameter:
                where: 操作がどのデータセットで動作するかを指定するための'dna'または'pep'
        Return:
                Dataオブジェクト（変換なし）

### FastqParser.save(where=None, fmt=None)
        各Dataサンプルについて、'where'で指定されたデータセットを保存します。結果はconfig.pyで指定されたParser出力フォルダにファイルとして書き込まれます。

        Parameter:
                where: 操作がどのデータセットで動作するかを指定するための'dna'または'pep'
                fmt: 出力ファイルの形式。サポートされている値は'npy'、'fasta'、'csv'
        Return:
        Dataオブジェクト（変換なし）
