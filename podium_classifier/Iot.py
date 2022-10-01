class Iot:

    def __init__(self, skAcctKey=None, Transactioncode=None, TransactionDate=None, TransactionID=None, OriginatorBankCode=None, BeneficiaryBankCode=None,
                 TransactionTypeCode=None, CardNumber=None, OriginalAmount=None, CurrencyCode=None, Amount=None, TransactionDescription=None,
                 skTransferAcctKey=None, ChannelCode=None, CashierIndicator=None, BranchIndicator=None, ATMIndicator=None, PhoneIndicator=None,
                 eBankingIndicator=None, MobileBankingIndicator=None, ContactlessIndicator=None, QuickPayIndicator=None,
                 InterbanktransferIndicator=None, StandingOrderIndicator=None, TransferCountry=None, MerchantName=None, MechantCode=None,
                 Debit_CreditIndicator=None, MCCCodeID=None, CardPresentIndicator=None, CardInstalIndicator=None, TransferskAcctKeyNACE=None,TxnCode=None,
                 Category=None, Master_Category=None, step=None, beneficiary_skCIF=None, customer_skCIF=None, ):

        self.skAcctKey = str(skAcctKey)
        self.Transactioncode = str(Transactioncode)
        self.TransactionDate = str(TransactionDate)
        self.TransactionID = str(TransactionID)
        self.OriginatorBankCode = str(OriginatorBankCode)
        self.BeneficiaryBankCode = str(BeneficiaryBankCode)
        self.TransactionTypeCode = str(TransactionTypeCode)
        self.CardNumber = str(CardNumber)
        self.OriginalAmount = str(OriginalAmount)
        self.CurrencyCode = str(CurrencyCode)
        self.Amount = Amount
        self.TransactionDescription = str(TransactionDescription)
        self.skTransferAcctKey = str(skTransferAcctKey)
        self.ChannelCode = str(ChannelCode)
        self.CashierIndicator = str(CashierIndicator)
        self.BranchIndicator = str(BranchIndicator)
        self.ATMIndicator = str(ATMIndicator)
        self.PhoneIndicator = str(PhoneIndicator)
        self.eBankingIndicator = str(eBankingIndicator)
        self.MobileBankingIndicator = str(MobileBankingIndicator)
        self.ContactlessIndicator = str(ContactlessIndicator)
        self.QuickPayIndicator = str(QuickPayIndicator)
        self.InterbanktransferIndicator = str(InterbanktransferIndicator)
        self.StandingOrderIndicator = str(StandingOrderIndicator)
        self.TransferCountry = str(TransferCountry)
        self.MerchantName = str(MerchantName)
        self.MechantCode = str(MechantCode)
        self.Debit_CreditIndicator = str(Debit_CreditIndicator)
        self.MCCCodeID = str(MCCCodeID)
        self.CardPresentIndicator = str(CardPresentIndicator)
        self.CardInstalIndicator = str(CardInstalIndicator)
        self.TransferskAcctKeyNACE = str(TransferskAcctKeyNACE)
        self.Category = Category
        self.Master_Category = Master_Category
        self.step = step
        self.TxnCodeID = str(self.TransactionTypeCode) + self.Debit_CreditIndicator
        self.beneficiary_skCIF = str(beneficiary_skCIF)
        self.customer_skCIF = str(customer_skCIF)
