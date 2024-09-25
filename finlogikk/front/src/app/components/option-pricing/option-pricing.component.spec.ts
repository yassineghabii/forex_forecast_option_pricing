import { ComponentFixture, TestBed } from '@angular/core/testing';

import { OptionPricingComponent } from './option-pricing.component';

describe('OptionPricingComponent', () => {
  let component: OptionPricingComponent;
  let fixture: ComponentFixture<OptionPricingComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ OptionPricingComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(OptionPricingComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
