package WEKA;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Date;

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
            printWriter.println("BIGARREN ARGUMENTUA: ARFF FITXATEGIAK -> " + args[1]);
            printWriter.println("HIRUGARREN ARGUMENTUA: ESTIMAZIOAK EGITEKO TXT -> " + args[2] + "\n");
            datuakKargatu(args[1]);
            ariketaEgin(args[0]);
            printWriter.close();
            fileWriter.close();
        }
    }

    private static void ariketaEgin(String path) throws Exception {
        Classifier naiveBayes = (Classifier)SerializationHelper.read(path);
        Evaluation eval = new Evaluation(test);
        eval.evaluateModel(naiveBayes,test);
        printWriter.println("TEST-AK EBALUATU --> ");
        printWriter.println("ACCURACY -> " + eval.pctCorrect());
        printWriter.println("CONFUSION MATRIX -> " + eval.toMatrixString());
    }

    private static void datuakKargatu(String arffFtixategia) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(arffFtixategia);
        test = source.getDataSet();
        if(test.classIndex() == -1) test.setClassIndex(test.numAttributes()-1);
    }

}
