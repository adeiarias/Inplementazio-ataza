package WEKA;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.ZeroR;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Random;

public class AzterketarenEdukia {

    public static void main(String[] args) throws Exception {
        if(args.length == 2) {
            System.out.println("\nEZ DITUZU ATRIBUTUAK BEHAR BEZALA JARRI. HURRENGO ITXURA IZAN BEHARKO LUKE:\n" +
                    "1.ATRIBUTUA -> ARFF FITXATEGIAREN KOKAPENA\n" +
                    "2.ATRIBUTUA -> EMAITZAK GORDE NAHI DITUZUN FITXATEGIAREN KOKAPENA");
        }else{
            DataSource source = new DataSource(args[0]);
            Instances data = source.getDataSet();
            if(data.classIndex() == -1) data.setClassIndex(data.numAttributes()-1);

            FileWriter fileWriter = new FileWriter(new File(args[1]));
            PrintWriter printWriter = new PrintWriter(fileWriter);

            datuak(data,printWriter);
            filtroak(data);
            ereduSailkatzaileak(data, printWriter);
            ebaluazioMetrikak(data, printWriter);
            fileWriter.close();
        }
    }

    private static void ebaluazioMetrikak(Instances data, PrintWriter printWriter) throws Exception {
        //NaiveBayes eredu sailkatzailea eta k-fold cross validation ebaluazio metodoa erabilita, ebaluazio metrikak aterako dira
        NaiveBayes naiveBayes = new NaiveBayes();
        naiveBayes.buildClassifier(data);

        Evaluation evaluation = new Evaluation(data);
        evaluation.crossValidateModel(naiveBayes,data,5,new Random(1));
        printWriter.println("\nAccuracy -> " + evaluation.pctCorrect());
       /* printWriter.println("Precision -> " + evaluation.precision(data.numAttributes()-1));
        printWriter.println("Recall -> " + evaluation.recall(data.numAttributes()-1));
        printWriter.println("F-measure -> " + evaluation.fMeasure(data.numAttributes()-1));*/
        printWriter.println("Confusion matrix -> \n" + evaluation.toMatrixString());
        printWriter.println("ESTATISTIKA OSOAK -> " + evaluation.toSummaryString());
    }

    private static void ereduSailkatzaileak(Instances data, PrintWriter printWriter) throws Exception {
        //EREDU SAILKATZAILE BAKOITZEKO HOLD-OUT, EBALUAZIO EZ-ZINTZOA ETA CROSS VALIDATION
        //ZeroR
        ZeroR zeroR = new ZeroR();
        zeroR.buildClassifier(data);
        ebaluazioakAtera(zeroR,data, printWriter,"ZeroR ebaluazioa");

        //OneR
        OneR oneR = new OneR();
        oneR.buildClassifier(data);
        ebaluazioakAtera(oneR,data, printWriter,"OneR ebaluazioa");

        //IBK
        IBk iBk = new IBk();
        iBk.buildClassifier(data);
        ebaluazioakAtera(iBk,data,printWriter,"iBk ebaluazioa");

        //NaiveBayes
        NaiveBayes naiveBayes = new NaiveBayes();
        naiveBayes.buildClassifier(data);
        ebaluazioakAtera(naiveBayes,data,printWriter,"NaiveBayes ebaluazioa");
    }

    private static void ebaluazioakAtera(Classifier classifier, Instances data, PrintWriter fitxategia,String mota) throws Exception{
        fitxategia.println("\n"+mota+":");
        fitxategia.println("EBALUAZIO EZ-ZINTZOA");
        Evaluation evalEzZintzoa = new Evaluation(data);
        evalEzZintzoa.evaluateModel(classifier,data);
        fitxategia.println("ACCURACY -> " + evalEzZintzoa.pctCorrect()+"\n");

        fitxategia.println("HOLD-OUT");
        Randomize randomize = new Randomize();
        randomize.setInputFormat(data);
        Instances random = Filter.useFilter(data,randomize);

        RemovePercentage removePercentageTrain = new RemovePercentage();
        removePercentageTrain.setInputFormat(random);
        removePercentageTrain.setInvertSelection(true);
        removePercentageTrain.setPercentage(70.0);
        Instances train = Filter.useFilter(random,removePercentageTrain);

        RemovePercentage removePercentageTest = new RemovePercentage();
        removePercentageTest.setInputFormat(random);
        removePercentageTest.setInvertSelection(false);
        removePercentageTest.setPercentage(70.0);
        Instances test = Filter.useFilter(random,removePercentageTest);

        Evaluation holdOut = new Evaluation(train);
        holdOut.evaluateModel(classifier,test);
        fitxategia.println("ACCURACY -> " + holdOut.pctCorrect()+"\n");

        fitxategia.println("K-FOLD CROSS VALIDATION");
        Evaluation crossValidation = new Evaluation(data);
        crossValidation.crossValidateModel(classifier,data,5,new Random(1));
        fitxategia.println("ACCURACY -> " + crossValidation.pctCorrect()+"\n");
    }

    private static void filtroak(Instances data) throws Exception{
        //RANDOMIZE
        Randomize randomize = new Randomize();
        randomize.setInputFormat(data);
        Instances random = Filter.useFilter(data,randomize);

        //DATU SORTA PARTIKETA
        //SORTAREN LEHENENG0 %70-A LORTU
        RemovePercentage removePercentageTrain = new RemovePercentage();
        removePercentageTrain.setInputFormat(random);
        removePercentageTrain.setPercentage(70.0);
        removePercentageTrain.setInvertSelection(true);
        Instances train = Filter.useFilter(random,removePercentageTrain);

        //SORTAREN AZKENEKO %30-A LORTU
        RemovePercentage removePercentageTest = new RemovePercentage();
        removePercentageTest.setInputFormat(random);
        removePercentageTest.setInvertSelection(false);
        removePercentageTest.setPercentage(70.0);
        Instances test = Filter.useFilter(random,removePercentageTest);

        //ATRIBUTUEN HAUTAPENA

    }

    private static void datuak(Instances data, PrintWriter printWriter) {
        printWriter.println("INSTANTZIA KOPURUA -> " + data.numInstances());
        printWriter.println("ATRIBUTU KOPURUA -> " + data.numAttributes());
        printWriter.println("LEHENENGO ATRIBUTUAREN MOTA -> " + data.attribute(0).type() + "\n");

        printWriter.println("ATRIBUTU NUMERIKOAREN EZAUGARRIAK:");
        printWriter.println("MINIMOA -> " + data.attributeStats(0).numericStats.min);
        printWriter.println("MAXIMOA -> " + data.attributeStats(0).numericStats.max);
        printWriter.println("BATAZBESTEKOA -> " + data.attributeStats(0).numericStats.mean);
        printWriter.println("DESBIDERAZIOA -> " + data.attributeStats(0).numericStats.stdDev + "\n");

        printWriter.println("ATRIBUTU NOMINALEN EZAUGARRIAK:");
        int[] counts = data.attributeStats(1).nominalCounts;
        int min = 0; //HEMEN ATRIBUTU MINIMOAREN POSIZIOA GORDEKO DA, GERO BALIOA ETA LABELA ERRAZ LORTZEKO
        for(int i=0; i<counts.length; i++){
            if(counts[min] > counts[i]) min = i;
            printWriter.println(data.attribute(1).value(i) + " -> " + counts[i] + " | MAIZTASUNA -> " + (float)counts[i]/data.attributeStats(1).totalCount);
        }
        printWriter.println("BALIO MINIMOA: " + data.attribute(1).value(min) + " -> " + counts[min] + "\n");

        printWriter.println("LEHENENGO ATRIBUTUAREN MISSING VALUES -> " + data.attributeStats(0).missingCount);
        printWriter.println("LEHENENGO ATRIBUTUAREN UNIQUE VALUES -> " + data.attributeStats(0).uniqueCount);
        printWriter.println("LEHENENGO ATRIBUTUAREN DISTINCT VALUES -> " + data.attributeStats(0).distinctCount);

    }
}