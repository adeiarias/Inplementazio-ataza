package WEKA;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Prediction;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils;

import javax.print.attribute.standard.MediaSize;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Date;
import java.util.List;

public class PRAKTIKA4_ARIKETA2 {
    private static Instances test;
    private static PrintWriter printWriter;

    public static void main(String[] args) throws Exception {
        if(args.length != 3) {
            System.out.println("EZ DITUZU BEHAR DIREN PARAMETROAK SARTU");
            System.out.println("HURRENGO MODUAN JARRI BEHARKO DUZU:");
            System.out.println("java -jar IragarpenakEgin.jar  /paht/to/karpeta/NB.model  /paht/to/test_blind.arff  /paht/to/irteerako/karpeta/test_predictions.txt");
        }else{
            FileWriter fileWriter = new FileWriter(args[2]);
            printWriter = new PrintWriter(fileWriter);
            printWriter.println("EGIKARITUTAKO DATA -> " + new Date());
            printWriter.println("LEHENENGO ARGUMENTUA: NAIVEBAYES MODELOA KARGATZEKO HELBIDEA -> " + args[0]);
            printWriter.println("BIGARREN ARGUMENTUA: ARFF FITXATEGIA -> " + args[1]);
            printWriter.println("HIRUGARREN ARGUMENTUA: ESTIMAZIOKO DATUAK -> " + args[2] + "\n");
            datuakKargatu(args[0]);
            ariketaEgin(args[1]);
            printWriter.close();
            fileWriter.close();
        }
    }

    private static void ariketaEgin(String path) throws Exception {
        //Classifier naiveBayes = (Classifier)SerializationHelper.read(path);
        NaiveBayes naiveBayes = new NaiveBayes();
        naiveBayes.buildClassifier(test);
        Evaluation eval = new Evaluation(test);
        eval.evaluateModel(naiveBayes,test);
        printWriter.println("TEST-AK EBALUATU --> ");
        printWriter.println("ACCURACY -> " + eval.pctCorrect());
        printWriter.println("CONFUSION MATRIX -> " + eval.toMatrixString());
        printWriter.println("\nOrain ereduak iragarritako balioak, balio errealekin konparatuko ditugu");
        printWriter.println("EREDU IGARARLEA BALIO ERREALAREKIN BAT EZ EGINEZ GERO X BAT AGERTUKO DA\n");
        estimazioakAtera(eval);
        estimazioakAteraIndividualmente(naiveBayes);
    }

    private static void estimazioakAteraIndividualmente(Classifier eredua) throws Exception{
        printWriter.println("\nPREDIKZIOAK EGIN EBALUAZIOA GABE...");
        for(Instance i : test){
            double pred = eredua.classifyInstance(i);
            printWriter.println(", actual: " + test.classAttribute().value((int) i.classValue()));
            printWriter.println(", predicted: " + test.classAttribute().value((int) pred));
        }
    }

    private static void estimazioakAtera(Evaluation estimazioak) {
        printWriter.println("PREDIKZIOAK EGITEN HASIKO GARA...");
        //Lehenik eta behin, predictions arraylist-arekin egingo dugu ariketa
        List<Prediction> lista = estimazioak.predictions();
        int i=0;
        int erreala,estimazioa;
        String erre,esti;
        for(Prediction pred : lista){
            erreala = (int)pred.actual();
            estimazioa = (int)pred.predicted();
            erre = test.attribute(test.classIndex()).value(erreala);
            esti = test.attribute(test.classIndex()).value(estimazioa);
            if(erreala != estimazioa){
                printWriter.println(i + " posizian -> BALIO ERREALA: " + erre + " ESTIMATUTAKO BALIOA: " + esti + " GAIZKI EGINDA? BAI");
            }else{
                printWriter.println(i + " posizian -> BALIO ERREALA: " + erre + " ESTIMATUTAKO BALIOA: " + esti + " GAIZKI EGINDA? EZ");
            }
            i++;
        }
    }

    private static void datuakKargatu(String arffFtixategia) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(arffFtixategia);
        test = source.getDataSet();
        if(test.classIndex() == -1) test.setClassIndex(test.numAttributes()-1);
    }

}
