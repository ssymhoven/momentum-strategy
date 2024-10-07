# Momentum

SPX: https://www.spglobal.com/spdji/en/documents/methodologies/methodology-sp-momentum-indices.pdf - S. 19

SXXP: https://www.stoxx.com/document/Indices/Common/Indexguide/stoxx_index_guide.pdf - S. 495


```
=BQL([@[bloomberg_query]]; "slope(#bmrk-0.015, #Px-0.015)";"#Px=Pct_diff(px_last(dates=range(-1y,0d))),#bmrk = 0.6 * value(#px,['SXXEWR Index']) + 0.4 * value(#px,['SPXEWNTR Index'])";"date=range(-1y,0d),Currency=EUR")
=BQL([@[bloomberg_query]]; "intercept(#bmrk, #Px)";"#Px=Pct_diff(px_last(dates=range(-1y,0d))),#bmrk = 0.6 * value(#px,['SXXEWR Index']) + 0.4 * value(#px,['SPXEWNTR Index'])";"date=range(-1y,0d),Currency=EUR")
```